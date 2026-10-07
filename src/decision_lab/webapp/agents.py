"""Agent registry for the web UI: one decide() interface over 3 models."""

import time
from dataclasses import dataclass

import torch

from decision_lab.config import Config
from decision_lab.head.model import (
    TypedDecisionHead,
    build_question_spec,
    create_random_head,
    get_device,
    predict_all,
)
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
        name="Trained Typed Head",
        description="Frozen Qwen3.5 backbone + trained typed decision head "
                    "(choice/score/noul in one forward pass, calibrated confidence).",
    ),
    AgentInfo(
        agent_id="head_random",
        name="Random Typed Head",
        description="Same backbone, but an untrained (randomly initialized) typed head.",
    ),
    AgentInfo(
        agent_id="prompt_lm_zero_shot",
        name="Prompt LM (zero-shot)",
        description="The backbone prompted via chat to answer every question in text.",
    ),
)


class TypedHeadAgent:
    """Answers the full question bank via live embedding → head forward pass."""

    system_prompt = None   # no textual prompt: the state is embedded directly

    def __init__(self, head: TypedDecisionHead, server, agent_id: str, temperatures: dict | None = None):
        self._head = head
        self._server = server
        self.agent_id = agent_id
        self._temperatures = temperatures or {}
        self._device = get_device()
        self._head.eval()
        self._head.to(self._device)

    def decide(self, state) -> tuple[dict, str, float]:
        """Return ({qid: decoded answer dict}, raw_output, latency_ms)."""
        t0 = time.perf_counter()
        [embedding] = self._server.embed([state.render()])
        feat = torch.tensor([embedding], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            outputs = self._head(feat)
        decoded = predict_all(self._head.question_spec, outputs, self._temperatures)
        latency_ms = (time.perf_counter() - t0) * 1000
        return decoded, "", latency_ms


class PromptAgentAdapter:
    """Adapts PromptAgent (answers, raw) to the simulator's (answers, raw, latency)."""

    def __init__(self, agent: PromptAgent, agent_id: str = "prompt_lm_zero_shot"):
        self._agent = agent
        self.agent_id = agent_id

    @property
    def system_prompt(self) -> str:
        return self._agent.system_prompt

    def decide(self, state) -> tuple[dict, str, float]:
        """Return ({qid: predicted value or missing if unparseable}, raw, latency)."""
        t0 = time.perf_counter()
        answers, raw = self._agent.decide(state)
        latency_ms = (time.perf_counter() - t0) * 1000
        return answers, raw, latency_ms


def build_agents(cfg: Config, server) -> dict[str, object]:
    """Instantiate the 3 selectable agents from config + a running LlamaServer."""
    agents: dict[str, object] = {}

    trained, temperatures = _load_trained_head(cfg)
    agents["head_trained"] = TypedHeadAgent(trained, server, "head_trained", temperatures)
    agents["head_random"] = TypedHeadAgent(
        create_random_head(build_question_spec(cfg.questions),
                           hidden_dim=cfg.head.hidden_dim, dropout=cfg.head.dropout),
        server, "head_random",
    )
    agents["prompt_lm_zero_shot"] = PromptAgentAdapter(
        PromptAgent(cfg, server, mode=PromptMode.ZERO_SHOT)
    )
    return agents


def _load_trained_head(cfg: Config) -> tuple[TypedDecisionHead, dict]:
    """Load the trained checkpoint (state + temperatures + question spec)."""
    from decision_lab.head.model import load_head
    from decision_lab import MODELS_DIR

    path = MODELS_DIR / "head_trained.pt"
    try:
        head = load_head(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {path} — retrain with `python -m decision_lab train` "
            f"(the file may be an old gridworld checkpoint). Original error: {exc}"
        ) from exc

    spec = build_question_spec(cfg.questions)
    if head.question_spec != spec:
        print("  warning: checkpoint question bank differs from config; "
              "using checkpoint spec for inference")
    return head, head.temperatures