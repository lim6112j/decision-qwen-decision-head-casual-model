"""FastAPI app: SSE-streamed typed-question evaluation with selectable models."""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from decision_lab import CONFIG_DIR, DATA_DIR, PROJECT_ROOT
from decision_lab.backbone.llama_server import LlamaServer
from decision_lab.config import Config, load_config
from decision_lab.head.model import build_question_spec
from decision_lab.real.labels import (
    SOURCE_ACCEPTED,
    SOURCE_AUTO,
    SOURCE_OVERRIDDEN,
    LabelItem,
    append_label,
    build_queue,
    load_labels,
    stats as label_stats,
)
from decision_lab.real.teacher import LabelingError, OpenRouterLabeler
from decision_lab.states.dataset import TextState, load_dataset
from decision_lab.webapp.agents import AGENT_INFOS, DynamicHeadAgent, build_agents
from decision_lab.webapp.simulator import evaluate_agent
from decision_lab.webapp.traffic_log import TrafficLogger

STATIC_DIR = PROJECT_ROOT / "web" / "static"
TEXT_PREVIEW_CHARS = 160
TRAFFIC_DIR = DATA_DIR / "traffic"
REAL_DIR = DATA_DIR / "real"
LABELS_PATH = REAL_DIR / "labels.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WebAppState:
    """Holds config, llama-server, the pre-generated states, and the 3 agents."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.server: Optional[LlamaServer] = None
        self.agents = {}
        self.states: list[TextState] = []
        self.question_spec: dict = {}
        self.run_lock = asyncio.Lock()
        self.traffic = TrafficLogger(TRAFFIC_DIR, enabled=cfg.web.log_traffic)

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
    # Honor the config selected by `python -m decision_lab --config … ui`
    # (cmd_ui exports DECISION_LAB_CONFIG); default config otherwise.
    cfg_path = os.environ.get("DECISION_LAB_CONFIG", str(CONFIG_DIR / "default.yaml"))
    cfg = load_config(cfg_path)
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
    custom_fields: Optional[list[str]] = None
    """Optional caller-controlled field chunking of custom_text (v2 head).
    Used verbatim — the caller owns the chunking, so it must match how the
    head was trained (heuristic split via states/fields.py is the default
    when omitted). Ignored for legacy v1 checkpoints (pooled state)."""
    questions: list[dict] = Field(min_length=1)
    """Each dict: {"type": "choice", "options": [...], "question": "..."}
       or {"type": "score", "levels": [...], "question": "..."}
       or {"type": "noul", "question": "..."}
    The "question" string conditions the head (v3 question-fused queries) —
    it may be omitted, in which case a learned null-question behavior is
    used. Legacy (pre-v3) checkpoints ignore it."""


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
async def decide_dynamic(req: DynamicDecideRequest, request: Request):
    """Evaluate a state against fully dynamic question configs.

    Uses the trained dynamic head agent. Returns decoded answers for each
    question in the request — option/level labels determine the output space.

    Opt-in capture: when ``web.log_traffic`` is enabled and the caller sends
    ``X-Decision-Lab-Log: 1``, the call is written to data/traffic/ for the
    real-data labeling pipeline (see src/decision_lab/real/).
    """
    if state is None:
        raise HTTPException(status_code=503, detail="server not ready")

    dynamic_agent = state.agents.get("head_dynamic")
    if dynamic_agent is None or not isinstance(dynamic_agent, DynamicHeadAgent):
        raise HTTPException(
            status_code=400,
            detail="dynamic head agent not available; train with `python -m decision_lab train-dynamic`",
        )

    _validate_dynamic_questions(req.questions)

    if req.custom_fields and not req.custom_text:
        raise HTTPException(
            status_code=400,
            detail="custom_fields requires custom_text (it chunks the custom text)",
        )
    s = _resolve_state_dynamic(req)
    if req.custom_fields and not dynamic_agent._head.state_set:
        raise HTTPException(
            status_code=400,
            detail="custom_fields requires a v2 (state_set) checkpoint — retrain the dynamic head",
        )
    try:
        answers, latency_ms = dynamic_agent.decide_dynamic(s, req.questions, fields=req.custom_fields)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"dynamic decision failed: {exc}") from exc

    _maybe_log_traffic(request, s, req, answers, latency_ms)

    return {
        "state": _state_payload(s),
        "answers": answers,
        "latency_ms": round(latency_ms, 1),
    }


def _maybe_log_traffic(request: Request, s: TextState, req: DynamicDecideRequest,
                       answers: list, latency_ms: float) -> None:
    """Write the call to the traffic log iff capture is enabled and opted in."""
    if state is None or not state.traffic.should_log(request.headers):
        return
    state.traffic.record({
        "ts": _now_iso(),
        "call_id": uuid.uuid4().hex,
        "text": s.text,
        "fields": req.custom_fields,
        "doc_id": s.doc_id,
        "state_type": s.state_type,
        "questions": req.questions,
        "answers": answers,
        "latency_ms": round(latency_ms, 1),
    })


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


def _validate_dynamic_questions(questions: list[dict]) -> None:
    """Fail fast with clear 400s on malformed question configs.

    Without this, a missing options/levels key surfaces as a 500 KeyError
    from deep inside the head — useless to the caller.
    """
    for i, q in enumerate(questions):
        where = f"questions[{i}]"
        if not isinstance(q, dict):
            raise HTTPException(status_code=400, detail=f"{where} must be an object")
        kind = q.get("type")
        if kind not in ("choice", "score", "noul"):
            raise HTTPException(
                status_code=400,
                detail=f"{where}.type must be 'choice' | 'score' | 'noul', got {kind!r}",
            )
        if kind == "choice" and not q.get("options"):
            raise HTTPException(
                status_code=400,
                detail=f"{where} is type=choice but has no non-empty 'options' list",
            )
        if kind == "score" and not q.get("levels"):
            raise HTTPException(
                status_code=400,
                detail=f"{where} is type=score but has no non-empty 'levels' list",
            )


# ---------------------------------------------------------------------------
# Real-traffic labeling
# ---------------------------------------------------------------------------


class LabelRequest(BaseModel):
    item_id: str = Field(min_length=1)
    gold_idx: int = Field(ge=0)
    source: str


class AutoLabelRequest(BaseModel):
    item_id: Optional[str] = None   # None → the first pending item


def _teacher_model() -> str:
    if state is not None:
        return state.cfg.real.teacher_model
    return load_config(CONFIG_DIR / "default.yaml").real.teacher_model


def _label_item_payload(item: LabelItem) -> dict:
    return {
        "item_id": item.item_id,
        "call_id": item.call_id,
        "text": item.text,
        "fields": item.fields,
        "state_type": item.state_type,
        "question": item.question,
        "kind": item.kind,
        "options": item.options,
        "predicted_idx": item.predicted_idx,
        "gold_idx": item.gold_idx,
        "gold_label": item.gold_label(),
        "source": item.source,
        "labeled_by": item.labeled_by,
    }


def _find_pending(item_id: str) -> LabelItem:
    for item in build_queue(TRAFFIC_DIR, LABELS_PATH):
        if item.item_id == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"pending item {item_id} not found")


@app.get("/api/label/queue")
def label_queue():
    """Every pending (unlabeled) item, pre-filled with the head's prediction."""
    return {"items": [_label_item_payload(i) for i in build_queue(TRAFFIC_DIR, LABELS_PATH)]}


@app.get("/api/label/stats")
def label_stats_endpoint():
    return label_stats(TRAFFIC_DIR, LABELS_PATH)


@app.post("/api/label")
def submit_label(req: LabelRequest):
    """Record a manual label (accept the pre-fill or override it)."""
    if req.source not in (SOURCE_ACCEPTED, SOURCE_OVERRIDDEN):
        raise HTTPException(
            status_code=400,
            detail=f"source must be '{SOURCE_ACCEPTED}' or '{SOURCE_OVERRIDDEN}'",
        )
    # Idempotent: re-labeling an already-labeled item is a no-op, not a 404.
    already = load_labels(LABELS_PATH)
    if req.item_id in already:
        return {"written": False, "item": _label_item_payload(already[req.item_id])}
    item = _find_pending(req.item_id)
    if req.gold_idx >= len(item.options):
        raise HTTPException(
            status_code=400,
            detail=f"gold_idx {req.gold_idx} out of range for {len(item.options)} options",
        )
    item.gold_idx = req.gold_idx
    item.source = req.source
    item.labeled_by = "human"
    written = append_label(item, LABELS_PATH)
    return {"written": written, "item": _label_item_payload(item)}


def _auto_label(item: LabelItem) -> LabelItem:
    """Label one item with OpenRouter; raises LabelingError on failure."""
    labeler = OpenRouterLabeler(model=_teacher_model())
    item.gold_idx = labeler.label(item)
    item.source = SOURCE_AUTO
    item.labeled_by = f"openrouter:{labeler.model}"
    append_label(item, LABELS_PATH)
    return item


@app.post("/api/label/auto")
def auto_label_one(req: AutoLabelRequest):
    """Label one pending item with the OpenRouter model."""
    if req.item_id:
        item = _find_pending(req.item_id)
    else:
        queue = build_queue(TRAFFIC_DIR, LABELS_PATH)
        if not queue:
            raise HTTPException(status_code=404, detail="no pending items to label")
        item = queue[0]
    try:
        _auto_label(item)
    except LabelingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"item": _label_item_payload(item)}


@app.post("/api/label/auto-all")
async def auto_label_all():
    """Stream-label every pending item via OpenRouter (SSE).

    Emits one ``progress`` event per item (``ok`` with the labeled item, or
    ``error``) and a final ``done`` summary. Per-item failures are skipped
    and counted — one bad response never aborts the batch.
    """
    async def generate():
        try:
            labeler = OpenRouterLabeler(model=_teacher_model())
        except LabelingError as exc:
            yield _sse("fatal", {"detail": str(exc)})
            return

        queue = build_queue(TRAFFIC_DIR, LABELS_PATH)
        labeled = failed = 0
        for item in queue:
            try:
                item.gold_idx = labeler.label(item)
                item.source = SOURCE_AUTO
                item.labeled_by = f"openrouter:{labeler.model}"
                append_label(item, LABELS_PATH)
                labeled += 1
                yield _sse("progress", {"ok": True, "item": _label_item_payload(item)})
            except LabelingError as exc:
                failed += 1
                yield _sse("progress", {"ok": False, "item_id": item.item_id, "detail": str(exc)})
        yield _sse("done", {"labeled": labeled, "failed": failed,
                            "remaining": len(queue) - labeled})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")