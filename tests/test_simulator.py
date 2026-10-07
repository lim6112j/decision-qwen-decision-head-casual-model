"""Tests for the single-state evaluation engine (scripted agents)."""

from decision_lab.head.model import is_correct
from decision_lab.states.dataset import TextState
from decision_lab.webapp.simulator import evaluate_agent

QUESTION_SPEC = {
    "sentiment": {
        "type": "choice",
        "options": ["positive", "negative", "neutral"],
        "option_descriptions": {},
    },
    "quality": {"type": "score", "levels": ["Poor", "Fair", "Good", "Excellent"]},
    "is_urgent": {"type": "noul", "question": "…"},
}

GOLD = {"sentiment": "negative", "quality": 1, "is_urgent": True}
STATE = TextState(doc_id=1, state_type="email", text="Angry email!", labels=dict(GOLD))


class ScriptedHeadAgent:
    """Returns decoded-answer dicts (like TypedHeadAgent)."""

    def __init__(self, answers, latency=5.0):
        self._answers = answers
        self._latency = latency

    def decide(self, state):
        decoded = {
            "sentiment": {
                "predicted": self._answers["sentiment"],
                "distribution": {"negative": 0.8, "neutral": 0.1, "positive": 0.1},
                "confidence": 0.8,
            },
            "quality": {
                "predicted": self._answers["quality"],
                "expected": float(self._answers["quality"]),
                "distribution": {"Poor": 0.1, "Fair": 0.8, "Good": 0.1, "Excellent": 0.0},
                "confidence": 0.8,
            },
            "is_urgent": {
                "predicted": self._answers["is_urgent"],
                "distribution": {"true": 0.9, "false": 0.1},
                "confidence": 0.9,
            },
        }
        return decoded, "", self._latency


class TestEvaluateAgent:
    def test_head_agent_scoring(self):
        out = evaluate_agent("a", "Agent A", STATE,
                             ScriptedHeadAgent(GOLD).decide, QUESTION_SPEC)
        assert out.mean_accuracy == 1.0
        assert out.parse_failures == 0
        assert all(q.correct for q in out.questions)
        assert out.questions[0].distribution is not None
        assert out.questions[0].confidence == 0.8

    def test_wrong_answers_scored_against_gold(self):
        wrong = {"sentiment": "positive", "quality": 3, "is_urgent": False}
        out = evaluate_agent("a", "Agent A", STATE,
                             ScriptedHeadAgent(wrong).decide, QUESTION_SPEC)
        assert out.mean_accuracy == 0.0

    def test_prompt_agent_missing_questions(self):
        def decide(state):
            return {"sentiment": "negative"}, "raw text", 1.0

        out = evaluate_agent("p", "Prompt", STATE, decide, QUESTION_SPEC)
        assert out.parse_failures == 2
        assert abs(out.mean_accuracy - 1 / 3) < 1e-9
        # prompt answers carry no distribution/confidence
        sentiment = next(q for q in out.questions if q.question_id == "sentiment")
        assert sentiment.distribution is None
        assert sentiment.confidence is None

    def test_custom_state_has_no_gold(self):
        custom = TextState(doc_id=-1, state_type="custom", text="hello", labels={})
        out = evaluate_agent("a", "Agent A", custom,
                             ScriptedHeadAgent(GOLD).decide, QUESTION_SPEC)
        assert out.mean_accuracy is None
        assert all(q.correct is None for q in out.questions)
        assert all(q.gold is None for q in out.questions)


def test_is_correct_none_is_wrong():
    spec = {"type": "noul"}
    assert not is_correct(spec, None, True)
    assert is_correct(spec, False, False)