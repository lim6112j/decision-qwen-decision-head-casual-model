"""Agent registry for the web simulator: one decide() interface over 3 models."""

import time
from dataclasses import dataclass

import torch

from decision_lab.config import Config
from decision_lab.env.gridworld import GridState
from decision_lab.head.model import DecisionHead, get_device
from decision_lab.prompt_lm.agent import PromptAgent, PromptMode


@dataclass
class AgentInfo:
    """Metadata for the UI's model selector."""

    agent_id: str
    name: str
    description: str


AGENT_INFOS: tuple[AgentInfo, ...] = (
    AgentInfo(
        agent_id="head_trained",
        name="Trained Head",
        description="Frozen Qwen3.5 backbone + trained MLP decision head (live embeddings).",
    ),
    AgentInfo(
        agent_id="head_random",
        name="Random Head",
        description="Same backbone, but an untrained (randomly initialized) MLP head.",
    ),
    AgentInfo(
        agent_id="prompt_lm_zero_shot",
        name="Prompt LM (zero-shot)",
        description="The backbone prompted via chat to answer 'Answer: <action>'.",
    ),
)


class HeadAgent:
    """Decides via live backbone embedding → MLP head forward pass."""

    system_prompt = None   # no textual prompt: the state is embedded directly

    def __init__(self, head: DecisionHead, server, agent_id: str):
        self._head = head
        self._server = server
        self.agent_id = agent_id
        self._device = get_device()
        self._head.eval()
        self._head.to(self._device)

    def decide(self, state: GridState) -> tuple[int, str, float]:
        t0 = time.perf_counter()
        [embedding] = self._server.embed([state.render()])
        feat = torch.tensor([embedding], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            action = self._head(feat).argmax(dim=1).item()
        latency_ms = (time.perf_counter() - t0) * 1000
        return action, "", latency_ms


class PromptAgentAdapter:
    """Adapts PromptAgent (action, raw) to the simulator's (action, raw, latency)."""

    def __init__(self, agent: PromptAgent, agent_id: str = "prompt_lm_zero_shot"):
        self._agent = agent
        self.agent_id = agent_id

    @property
    def system_prompt(self) -> str:
        return self._agent.system_prompt

    def decide(self, state: GridState) -> tuple[int | None, str, float]:
        t0 = time.perf_counter()
        action, raw = self._agent.decide(state)
        latency_ms = (time.perf_counter() - t0) * 1000
        return action, raw, latency_ms


def build_agents(cfg: Config, server) -> dict[str, object]:
    """Instantiate the 3 selectable agents from config + a running LlamaServer."""
    heads = {}
    for agent_id, head in [
        ("head_trained", _load_trained_head(cfg)),
        ("head_random", DecisionHead(
            hidden_dim=cfg.head.hidden_dim,
            num_actions=cfg.head.num_actions,
            dropout=cfg.head.dropout,
        )),
    ]:
        heads[agent_id] = HeadAgent(head, server, agent_id)

    prompt = PromptAgentAdapter(
        PromptAgent(cfg, server, mode=PromptMode.ZERO_SHOT)
    )
    return {
        "head_trained": heads["head_trained"],
        "head_random": heads["head_random"],
        "prompt_lm_zero_shot": prompt,
    }


def _load_trained_head(cfg: Config) -> DecisionHead:
    from decision_lab.head.model import load_head
    from decision_lab import MODELS_DIR

    return load_head(
        str(MODELS_DIR / "head_trained.pt"),
        hidden_dim=cfg.head.hidden_dim,
        num_actions=cfg.head.num_actions,
        dropout=cfg.head.dropout,
    )
