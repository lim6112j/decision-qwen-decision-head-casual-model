"""Agent registry for the web UI: one decide() interface over 4 models."""

import time
from dataclasses import dataclass

import torch

from decision_lab.config import Config
from decision_lab.head.dynamic_model import (
    DynamicDecisionHead,
    create_random_dynamic_head,
    decode_dynamic_answer,
    get_device,
    load_dynamic_head,
    make_choice_question,
    make_noul_question,
    make_score_question,
    question_option_texts,
)
from decision_lab.head.model import (
    TypedDecisionHead,
    build_question_spec,
    create_random_head,
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
        agent_id="head_dynamic",
        name="Trained Dynamic Head",
        description="Attention-based head: option values are inputs, not architecture. "
                    "Handles any number of options at inference without retraining.",
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


class DynamicHeadAgent:
    """Answers dynamic questions via attention-based slot filling.

    Option values are passed as text, embedded by the backbone, and scored
    against the state via scaled dot-product attention. The same trained head
    handles any number of options without retraining.

    Falls back to the fixed question bank (from config) when no dynamic
    questions are provided.
    """

    system_prompt = None

    def __init__(
        self,
        head: DynamicDecisionHead,
        server,
        agent_id: str,
        question_spec: dict | None = None,
        temperature: float = 1.0,
    ):
        self._head = head
        self._server = server
        self.agent_id = agent_id
        self._temperature = temperature
        self._device = get_device()
        self._head.eval()
        self._head.to(self._device)
        # Fallback: use fixed question bank for backward compatibility
        self._question_spec = question_spec or {}
        # Option embedding cache: {option_text: tensor}
        self._option_cache: dict[str, torch.Tensor] = {}

    def _ensure_options_embedded(self, option_texts: list[str]) -> dict[str, torch.Tensor]:
        """Embed option texts via backbone, cache on this agent instance."""
        missing = [t for t in option_texts if t not in self._option_cache]
        if missing:
            embs = self._server.embed(missing)
            for text, emb in zip(missing, embs):
                self._option_cache[text] = torch.tensor(
                    emb, dtype=torch.float32, device=self._device
                )
        return {t: self._option_cache[t] for t in option_texts}

    def decide(self, state) -> tuple[dict, str, float]:
        """Return ({qid: decoded answer dict}, raw_output, latency_ms).

        Uses the fixed question bank from config as the default question set.
        """
        t0 = time.perf_counter()
        [state_emb] = self._server.embed([state.render()])
        state_tensor = torch.tensor([state_emb], dtype=torch.float32, device=self._device)

        results = {}
        with torch.no_grad():
            for qid, spec in self._question_spec.items():
                kind = spec["type"]
                if kind == "noul":
                    cache = self._ensure_options_embedded(["false", "true"])
                    opt_embs = torch.stack([cache["false"], cache["true"]])
                    scores = self._head.forward_choice(state_tensor, opt_embs)
                    results[qid] = decode_dynamic_answer(
                        make_noul_question(spec.get("question", qid)),
                        scores,
                        self._temperature,
                    )
                elif kind == "choice":
                    option_texts = spec["options"]
                    cache = self._ensure_options_embedded(option_texts)
                    opt_embs = torch.stack([cache[t] for t in option_texts])
                    scores = self._head.forward_choice(state_tensor, opt_embs)
                    results[qid] = decode_dynamic_answer(
                        make_choice_question(option_texts, qid),
                        scores,
                        self._temperature,
                    )
                elif kind == "score":
                    level_texts = spec["levels"]
                    cache = self._ensure_options_embedded(level_texts)
                    opt_embs = torch.stack([cache[t] for t in level_texts])
                    scores = self._head.forward_score(state_tensor, opt_embs)
                    results[qid] = decode_dynamic_answer(
                        make_score_question(level_texts, qid),
                        scores,
                        self._temperature,
                    )

        latency_ms = (time.perf_counter() - t0) * 1000
        return results, "", latency_ms

    def decide_dynamic(
        self,
        state,
        questions: list[dict],
    ) -> tuple[list[dict], float]:
        """Decide with fully dynamic questions (no fixed bank).

        Args:
            state: TextState to evaluate.
            questions: list of dynamic question configs from make_*_question().

        Returns:
            (list of decoded answer dicts, latency_ms).
        """
        t0 = time.perf_counter()
        [state_emb] = self._server.embed([state.render()])
        state_tensor = torch.tensor([state_emb], dtype=torch.float32, device=self._device)

        results = []
        with torch.no_grad():
            for q in questions:
                kind = q["type"]
                if kind == "noul":
                    cache = self._ensure_options_embedded(["false", "true"])
                    opt_embs = torch.stack([cache["false"], cache["true"]])
                    scores = self._head.forward_choice(state_tensor, opt_embs)
                    results.append(decode_dynamic_answer(q, scores, self._temperature))
                else:
                    option_texts = question_option_texts(q)
                    cache = self._ensure_options_embedded(option_texts)
                    opt_embs = torch.stack([cache[t] for t in option_texts])
                    if kind == "choice":
                        scores = self._head.forward_choice(state_tensor, opt_embs)
                    else:
                        scores = self._head.forward_score(state_tensor, opt_embs)
                    results.append(decode_dynamic_answer(q, scores, self._temperature))

        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms


def build_agents(cfg: Config, server) -> dict[str, object]:
    """Instantiate the 4 selectable agents from config + a running LlamaServer."""
    agents: dict[str, object] = {}

    trained, temperatures = _load_trained_head(cfg)
    question_spec = build_question_spec(cfg.questions)
    agents["head_trained"] = TypedHeadAgent(trained, server, "head_trained", temperatures)
    agents["head_random"] = TypedHeadAgent(
        create_random_head(question_spec,
                           hidden_dim=cfg.head.hidden_dim, dropout=cfg.head.dropout),
        server, "head_random",
    )
    agents["head_dynamic"] = _load_dynamic_agent(cfg, server, question_spec)
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


def _load_dynamic_agent(
    cfg: Config, server, question_spec: dict
) -> DynamicHeadAgent:
    """Load a DynamicHeadAgent (trained or fallback to random init)."""
    from decision_lab import MODELS_DIR

    path = MODELS_DIR / "head_dynamic.pt"
    question_spec_copy = dict(question_spec)

    try:
        head = load_dynamic_head(str(path))
        temperature = getattr(head, "temperature", 1.0)
        print(f"  loaded dynamic head from {path} (T={temperature:.3f})")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"  no dynamic head checkpoint at {path} ({exc}); using random init")
        head = create_random_dynamic_head(
            hidden_dim=cfg.dynamic_head.hidden_dim,
            d_k=cfg.dynamic_head.d_k,
            dropout=cfg.dynamic_head.dropout,
        )
        temperature = 1.0

    return DynamicHeadAgent(
        head, server, "head_dynamic",
        question_spec=question_spec_copy,
        temperature=temperature,
    )