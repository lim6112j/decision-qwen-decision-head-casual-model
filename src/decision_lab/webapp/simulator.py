"""Single-state evaluation engine: run agents over a text state, one answer
per question. Framework-free so it can be unit-tested with scripted agents.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from decision_lab.head.model import is_correct

DecideFn = Callable[[Any], tuple[dict, str, float]]
"""Returns (answers, raw_output, latency_ms).

answers is either {qid: decoded answer dict} (head agents) or
{qid: predicted value} (prompt agents); missing qids are parse failures.
"""


@dataclass
class QuestionOutput:
    """One question's answer from one agent."""

    question_id: str
    question_type: str            # choice / score / noul
    predicted: Any                # option key / level index / bool; None if failed
    predicted_label: str          # display string
    distribution: Optional[dict[str, float]]   # {class label: probability}
    confidence: Optional[float]
    expected: Optional[float]     # score questions only (expected rubric level)
    gold: Any
    gold_label: str
    correct: Optional[bool]       # None when the agent gave no answer


@dataclass
class AgentOutput:
    """One agent's full result over a state."""

    agent_id: str
    agent_name: str
    latency_ms: float
    raw_output: str
    questions: list[QuestionOutput] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> Optional[float]:
        """Mean correctness over questions with gold labels; None if no gold."""
        judged = [q.correct for q in self.questions if q.correct is not None]
        return sum(judged) / len(judged) if judged else None

    @property
    def parse_failures(self) -> int:
        return sum(1 for q in self.questions if q.predicted is None)


def _predicted_label(spec_entry: dict, predicted) -> str:
    if predicted is None:
        return "—"
    kind = spec_entry["type"]
    if kind == "choice":
        return str(predicted)
    if kind == "score":
        levels = spec_entry["levels"]
        idx = min(int(predicted), len(levels) - 1)
        return f"{idx} ({levels[idx]})"
    return "true" if predicted else "false"


def _gold_label(spec_entry: dict, gold) -> str:
    if gold is None:
        return "—"
    kind = spec_entry["type"]
    if kind == "score":
        idx = min(int(gold), len(spec_entry["levels"]) - 1)
        return f"{idx} ({spec_entry['levels'][idx]})"
    if kind == "noul":
        return "true" if gold else "false"
    return str(gold)


def _normalize_answer(raw_answer) -> tuple[Optional[Any], Optional[dict], Optional[float], Optional[float]]:
    """Accept both decoded dicts (head) and bare values (prompt agent)."""
    if isinstance(raw_answer, dict) and "predicted" in raw_answer:
        return (
            raw_answer.get("predicted"),
            raw_answer.get("distribution"),
            raw_answer.get("confidence"),
            raw_answer.get("expected"),
        )
    return raw_answer, None, None, None


def evaluate_agent(
    agent_id: str,
    agent_name: str,
    state,
    decide_fn: DecideFn,
    question_spec: dict,
) -> AgentOutput:
    """Run one agent over one state; compare against gold labels when present."""
    answers, raw_output, latency_ms = decide_fn(state)

    questions = []
    for qid, spec_entry in question_spec.items():
        predicted, distribution, confidence, expected = _normalize_answer(answers.get(qid))
        gold = state.labels.get(qid) if state.labels else None
        questions.append(QuestionOutput(
            question_id=qid,
            question_type=spec_entry["type"],
            predicted=predicted,
            predicted_label=_predicted_label(spec_entry, predicted),
            distribution=distribution,
            confidence=confidence,
            expected=expected,
            gold=gold,
            gold_label=_gold_label(spec_entry, gold),
            correct=(
                is_correct(spec_entry, predicted, gold)
                if gold is not None
                else None
            ),
        ))

    return AgentOutput(
        agent_id=agent_id,
        agent_name=agent_name,
        latency_ms=latency_ms,
        raw_output=raw_output,
        questions=questions,
    )