"""FastAPI app: SSE-streamed typed-question evaluation with selectable models."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from random import Random
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from decision_lab import CONFIG_DIR, DATA_DIR, PROJECT_ROOT
from decision_lab.backbone.llama_server import LlamaServer
from decision_lab.config import Config, load_config
from decision_lab.head.model import build_question_spec
from decision_lab.states.dataset import TextState, load_dataset
from decision_lab.webapp.agents import AGENT_INFOS, DynamicHeadAgent, build_agents
from decision_lab.webapp.simulator import evaluate_agent

STATIC_DIR = PROJECT_ROOT / "web" / "static"
TEXT_PREVIEW_CHARS = 160


class WebAppState:
    """Holds config, llama-server, the pre-generated states, and the 3 agents."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.server: Optional[LlamaServer] = None
        self.agents = {}
        self.states: list[TextState] = []
        self.question_spec: dict = {}
        self.run_lock = asyncio.Lock()

    def start(self) -> None:
        gguf = Path(self.cfg.model.gguf_path).expanduser().resolve()
        self.server = LlamaServer(
            gguf,
            port=self.cfg.model.server_port,
            context_length=self.cfg.model.context_length,
        )
        self.server.start()
        self.agents = build_agents(self.cfg, self.server)
        self.question_spec = build_question_spec(self.cfg.questions)
        self.states = self._load_states()

    def stop(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None

    def _load_states(self) -> list[TextState]:
        """Sample pre-generated states from the in-distribution test set."""
        path = DATA_DIR / "test_indist.jsonl"
        if not path.exists():
            print(f"  warning: no states at {path}; UI will offer custom text only")
            return []
        states = load_dataset(path)
        rng = Random(self.cfg.web.random_seed)
        picked = rng.sample(states, min(self.cfg.web.num_states, len(states)))
        picked.sort(key=lambda s: s.doc_id)
        return picked


state: Optional[WebAppState] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    cfg = load_config(CONFIG_DIR / "default.yaml")
    state = WebAppState(cfg)
    state.start()
    yield
    state.stop()


app = FastAPI(title="Decision Lab — Typed Question Analysis", lifespan=lifespan)


class RunRequest(BaseModel):
    agent_id: str = Field(min_length=1)
    doc_id: Optional[int] = None      # index into state.states
    custom_text: Optional[str] = None  # overrides doc_id when provided


class DynamicDecideRequest(BaseModel):
    """Request to evaluate a state against fully dynamic question configs."""
    doc_id: Optional[int] = None
    custom_text: Optional[str] = None
    questions: list[dict] = Field(min_length=1)
    """Each dict: {"type": "choice", "options": [...], "question": "..."}
       or {"type": "score", "levels": [...], "question": "..."}
       or {"type": "noul", "question": "..."}"""


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _get_agent(agent_id: str):
    if state is None or agent_id not in state.agents:
        raise HTTPException(status_code=400, detail=f"unknown agent_id: {agent_id}")
    return state.agents[agent_id]


def _resolve_state(req: RunRequest) -> TextState:
    """Custom text wins; otherwise pick a pre-generated state by doc_id."""
    if req.custom_text:
        return TextState(doc_id=-1, state_type="custom", text=req.custom_text, labels={})
    if req.doc_id is None:
        raise HTTPException(status_code=400, detail="provide doc_id or custom_text")
    for s in state.states:
        if s.doc_id == req.doc_id:
            return s
    raise HTTPException(status_code=404, detail=f"doc_id {req.doc_id} not found")


def _state_payload(s: TextState) -> dict:
    return {
        "doc_id": s.doc_id,
        "state_type": s.state_type,
        "text": s.text,
        "has_gold": bool(s.labels),
    }


def _agent_output_payload(out) -> dict:
    d = asdict(out)
    d["latency_ms"] = round(out.latency_ms, 1)
    d["mean_accuracy"] = out.mean_accuracy
    d["parse_failures"] = out.parse_failures
    return d


def _run_stream(agent_id: str, s: TextState):
    """Generator yielding SSE events for a single agent over one state."""
    agent = _get_agent(agent_id)
    info = next(i for i in AGENT_INFOS if i.agent_id == agent_id)

    yield _sse("start", {
        "agent_id": agent_id,
        "agent_name": info.name,
        "state": _state_payload(s),
        "system_prompt": getattr(agent, "system_prompt", None),
    })

    out = evaluate_agent(agent_id, info.name, s, agent.decide, state.question_spec)
    yield _sse("result", _agent_output_payload(out))
    yield _sse("done", {"agent_id": agent_id})


@app.get("/api/agents")
def list_agents():
    return {"agents": [vars(info) for info in AGENT_INFOS]}


@app.get("/api/states")
def list_states():
    return {"states": [_state_payload(s) for s in state.states]}


@app.get("/api/questions")
def list_questions():
    """The fixed question bank, for the UI's result table."""
    return {"questions": state.question_spec}


@app.get("/api/status")
def status():
    return {"ready": state is not None and bool(state.agents)}


@app.post("/api/run")
async def run(req: RunRequest):
    _get_agent(req.agent_id)  # validate before opening the stream
    s = _resolve_state(req)

    async def generate():
        async with state.run_lock:
            for event in _run_stream(req.agent_id, s):
                yield event

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/compare")
async def compare(req: RunRequest):
    s = _resolve_state(req)

    async def generate():
        async with state.run_lock:
            for info in AGENT_INFOS:
                _get_agent(info.agent_id)
                yield _sse("agent_start", {"agent_id": info.agent_id, "name": info.name})
                for event in _run_stream(info.agent_id, s):
                    yield event

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/decide-dynamic")
async def decide_dynamic(req: DynamicDecideRequest):
    """Evaluate a state against fully dynamic question configs.

    Uses the trained dynamic head agent. Returns decoded answers for each
    question in the request — option/level labels determine the output space.
    """
    if state is None:
        raise HTTPException(status_code=503, detail="server not ready")

    dynamic_agent = state.agents.get("head_dynamic")
    if dynamic_agent is None or not isinstance(dynamic_agent, DynamicHeadAgent):
        raise HTTPException(
            status_code=400,
            detail="dynamic head agent not available; train with `python -m decision_lab train-dynamic`",
        )

    s = _resolve_state_dynamic(req)
    try:
        answers, latency_ms = dynamic_agent.decide_dynamic(s, req.questions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"dynamic decision failed: {exc}") from exc

    return {
        "state": _state_payload(s),
        "answers": answers,
        "latency_ms": round(latency_ms, 1),
    }


def _resolve_state_dynamic(req: DynamicDecideRequest) -> "TextState":
    """Resolve state for dynamic decision endpoint."""
    if req.custom_text:
        return TextState(doc_id=-1, state_type="custom", text=req.custom_text, labels={})
    if req.doc_id is None:
        raise HTTPException(status_code=400, detail="provide doc_id or custom_text")
    for s in (state.states if state else []):
        if s.doc_id == req.doc_id:
            return s
    raise HTTPException(status_code=404, detail=f"doc_id {req.doc_id} not found")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")