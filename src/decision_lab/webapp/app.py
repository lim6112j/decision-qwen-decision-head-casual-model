"""FastAPI app: SSE-streamed gridworld simulation with selectable models."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from random import Random
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from decision_lab import PROJECT_ROOT, CONFIG_DIR
from decision_lab.backbone.llama_server import LlamaServer
from decision_lab.config import Config, load_config
from decision_lab.env.gridworld import GridLayout, GridState
from decision_lab.webapp.agents import AGENT_INFOS, build_agents
from decision_lab.webapp.simulator import (
    EpisodeSummary,
    pick_start_pos,
    run_episode,
    summarize,
)

STATIC_DIR = PROJECT_ROOT / "web" / "static"


class WebAppState:
    """Holds config, llama-server, layouts, and the 3 agents."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.server: Optional[LlamaServer] = None
        self.agents = {}
        self.layouts: list[GridLayout] = []
        self.episode_lock = asyncio.Lock()

    def start(self) -> None:
        gguf = Path(self.cfg.model.gguf_path).expanduser().resolve()
        self.server = LlamaServer(
            gguf,
            port=self.cfg.model.server_port,
            context_length=self.cfg.model.context_length,
        )
        self.server.start()
        self.agents = build_agents(self.cfg, self.server)
        self.layouts = self._generate_layouts()

    def stop(self) -> None:
        if self.server:
            self.server.stop()
            self.server = None

    def _generate_layouts(self) -> list[GridLayout]:
        rng = Random(self.cfg.web.random_seed)
        g = self.cfg.grid
        layouts = []
        for _ in range(self.cfg.web.num_layouts):
            layouts.append(GridLayout.random(
                rows=g.size, cols=g.size,
                wall_density=g.wall_density,
                min_path_length=g.min_path_length, rng=rng,
            ))
        return layouts


state: Optional[WebAppState] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    cfg = load_config(CONFIG_DIR / "default.yaml")
    state = WebAppState(cfg)
    state.start()
    yield
    state.stop()


app = FastAPI(title="Decision Lab Simulator", lifespan=lifespan)


class RunRequest(BaseModel):
    agent_id: str = Field(min_length=1)
    layout_id: Optional[int] = None   # None → fresh random layout


class CompareRequest(BaseModel):
    layout_id: Optional[int] = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _resolve_layout(layout_id: Optional[int], rng: Random) -> tuple[GridLayout, tuple[int, int]]:
    """Return (layout, start_pos) for a layout id, or a fresh random layout."""
    if layout_id is None:
        layout = _random_layout(rng)
    else:
        if not 0 <= layout_id < len(state.layouts):
            raise HTTPException(status_code=404, detail=f"layout_id {layout_id} out of range")
        layout = state.layouts[layout_id]
    return layout, pick_start_pos(layout)


def _random_layout(rng: Random) -> GridLayout:
    g = state.cfg.grid
    return GridLayout.random(
        rows=g.size, cols=g.size,
        wall_density=g.wall_density,
        min_path_length=g.min_path_length, rng=rng,
    )


def _grid_payload(layout: GridLayout, agent_pos: tuple[int, int]) -> dict:
    return {
        "rows": layout.rows,
        "cols": layout.cols,
        "walls": [[int(layout.cells[r, c]) == 1 for c in range(layout.cols)]
                  for r in range(layout.rows)],
        "goal_pos": list(layout.goal_pos),
        "agent_pos": list(agent_pos),
    }


def _get_agent(agent_id: str):
    if state is None or agent_id not in state.agents:
        raise HTTPException(status_code=400, detail=f"unknown agent_id: {agent_id}")
    return state.agents[agent_id]


def _episode_stream(agent_id: str, layout_id: Optional[int]):
    """Generator yielding SSE events for a single episode run."""
    agent = _get_agent(agent_id)
    layout, start_pos = _resolve_layout(layout_id, Random())
    start_state = GridState(layout=layout, agent_pos=start_pos)
    optimal_steps = int(layout.bfs_distances()[start_pos])

    yield _sse("start", {
        "agent_id": agent_id,
        "grid": _grid_payload(layout, start_pos),
        "start_pos": list(start_pos),
        "optimal_steps": optimal_steps,
        "system_prompt": getattr(agent, "system_prompt", None),
    })

    steps = []
    for step in run_episode(layout, start_pos, agent.decide, state.cfg.web.max_steps):
        steps.append(step)
        yield _sse("step", {
            "step_idx": step.step_idx,
            "agent_pos": list(step.agent_pos),
            "new_pos": list(step.new_pos),
            "action": step.action,
            "action_name": step.action_name,
            "optimal_action": step.optimal_action,
            "optimal_action_name": step.optimal_action_name,
            "correct": step.correct,
            "latency_ms": round(step.latency_ms, 1),
            "raw_output": step.raw_output,
            "reached_goal": step.reached_goal,
            "state_text": step.state_text,
        })

    summary: EpisodeSummary = summarize(steps, optimal_steps)
    yield _sse("summary", {
        "agent_id": agent_id,
        "success": summary.success,
        "steps_used": summary.steps_used,
        "optimal_steps": summary.optimal_steps,
        "action_accuracy": round(summary.action_accuracy, 3),
        "parse_failures": summary.parse_failures,
    })


@app.get("/api/agents")
def list_agents():
    return {"agents": [vars(info) for info in AGENT_INFOS]}


@app.get("/api/layouts")
def list_layouts():
    return {
        "layouts": [
            {
                "layout_id": i,
                "rows": ly.rows,
                "cols": ly.cols,
                "optimal_from_start": int(ly.bfs_distances()[pick_start_pos(ly)]),
                "ascii": ly.to_text(pick_start_pos(ly)),
            }
            for i, ly in enumerate(state.layouts)
        ]
    }


@app.get("/api/status")
def status():
    return {"ready": state is not None and bool(state.agents)}


@app.post("/api/run")
async def run(req: RunRequest):
    _get_agent(req.agent_id)  # validate before opening the stream

    async def generate():
        # Hold the lock for the whole stream: the generator body runs
        # lazily after this handler returns.
        async with state.episode_lock:
            for event in _episode_stream(req.agent_id, req.layout_id):
                yield event

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/compare")
async def compare(req: CompareRequest):
    async def generate():
        async with state.episode_lock:
            for info in AGENT_INFOS:
                yield _sse("agent_start", {"agent_id": info.agent_id, "name": info.name})
                for event in _episode_stream(info.agent_id, req.layout_id):
                    yield event

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
