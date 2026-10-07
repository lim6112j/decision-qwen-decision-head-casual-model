"""Tests for the TypedDecisionHead (architecture, decoding, checkpointing)."""

import tempfile
from pathlib import Path

import pytest
import torch

from decision_lab.head.model import (
    TypedDecisionHead,
    create_random_head,
    decode_answer,
    get_device,
    is_correct,
    load_head,
    probabilities,
    predict_all,
    save_head,
)

QUESTION_SPEC = {
    "sentiment": {
        "type": "choice",
        "options": ["positive", "negative", "neutral"],
        "option_descriptions": {},
    },
    "quality": {"type": "score", "levels": ["Poor", "Fair", "Good", "Excellent"]},
    "is_urgent": {"type": "noul", "question": "Is this time-sensitive?"},
}


class TestTypedHeadArchitecture:
    def test_forward_returns_all_questions(self):
        model = TypedDecisionHead(question_spec=QUESTION_SPEC)
        x = torch.randn(4, 1024)
        outputs = model(x)
        assert set(outputs.keys()) == set(QUESTION_SPEC.keys())

    def test_forward_shapes(self):
        model = TypedDecisionHead(question_spec=QUESTION_SPEC, hidden_dim=64)
        outputs = model(torch.randn(8, 1024))
        assert outputs["sentiment"].shape == (8, 3)   # 3 options
        assert outputs["quality"].shape == (8, 4)     # 4 rubric levels
        assert outputs["is_urgent"].shape == (8, 2)   # binary

    def test_requires_question_spec(self):
        with pytest.raises(ValueError):
            TypedDecisionHead()

    def test_unknown_question_type(self):
        with pytest.raises(ValueError):
            TypedDecisionHead(question_spec={"bad": {"type": "trinary"}})


class TestProbabilities:
    def test_softmax_sums_to_one(self):
        model = TypedDecisionHead(question_spec=QUESTION_SPEC, dropout=0.0)
        model.eval()
        outputs = model(torch.randn(8, 1024))
        probs = probabilities(outputs)
        for qid, p in probs.items():
            sums = p.sum(dim=1)
            assert torch.allclose(sums, torch.ones(8), atol=1e-5), qid

    def test_noul_p_false_complements_p_true(self):
        model = TypedDecisionHead(question_spec=QUESTION_SPEC, dropout=0.0)
        model.eval()
        outputs = model(torch.randn(8, 1024))
        probs = probabilities(outputs)["is_urgent"]
        assert torch.allclose(probs[:, 0] + probs[:, 1], torch.ones(8), atol=1e-5)

    def test_temperature_preserves_argmax(self):
        torch.manual_seed(0)
        logits = torch.randn(16, 3) * 2
        p_hot = probabilities({"q": logits}, {"q": 0.1})["q"]
        p_cold = probabilities({"q": logits}, {"q": 10.0})["q"]
        assert torch.equal(p_hot.argmax(dim=1), p_cold.argmax(dim=1))
        # Sharper temperature concentrates mass
        assert p_hot.max(dim=1).values.mean() > p_cold.max(dim=1).values.mean()


class TestDecoding:
    def test_decode_choice(self):
        spec = {"type": "choice", "options": ["a", "b", "c"]}
        row = torch.tensor([0.1, 0.7, 0.2])
        d = decode_answer(spec, row)
        assert d["predicted"] == "b"
        assert abs(sum(d["distribution"].values()) - 1.0) < 1e-5
        assert d["confidence"] == d["distribution"]["b"]

    def test_decode_score_expected_level(self):
        spec = {"type": "score", "levels": ["Poor", "Fair", "Good", "Excellent"]}
        row = torch.tensor([0.0, 0.0, 0.5, 0.5])
        d = decode_answer(spec, row)
        assert d["predicted"] == 2
        assert abs(d["expected"] - 2.5) < 1e-6
        assert d["distribution"]["Excellent"] == 0.5

    def test_decode_noul(self):
        spec = {"type": "noul"}
        d_true = decode_answer(spec, torch.tensor([0.3, 0.7]))
        assert d_true["predicted"] is True
        assert d_true["distribution"]["true"] == pytest.approx(0.7, abs=1e-6)
        assert d_true["distribution"]["false"] == pytest.approx(0.3, abs=1e-6)
        assert d_true["distribution"]["true"] + d_true["distribution"]["false"] == pytest.approx(1.0)
        assert d_true["confidence"] == pytest.approx(0.7, abs=1e-6)

        d_false = decode_answer(spec, torch.tensor([0.8, 0.2]))
        assert d_false["predicted"] is False

    def test_predict_all_single_pass(self):
        model = TypedDecisionHead(question_spec=QUESTION_SPEC, dropout=0.0)
        model.eval()
        outputs = model(torch.randn(1, 1024))
        decoded = predict_all(QUESTION_SPEC, outputs, {"is_urgent": 2.0})
        assert set(decoded.keys()) == set(QUESTION_SPEC.keys())
        assert isinstance(decoded["is_urgent"]["predicted"], bool)
        assert decoded["sentiment"]["predicted"] in QUESTION_SPEC["sentiment"]["options"]


class TestIsCorrect:
    def test_choice_case_insensitive(self):
        spec = {"type": "choice", "options": ["Positive", "negative"]}
        assert is_correct(spec, "positive", "Positive")
        assert not is_correct(spec, "negative", "Positive")

    def test_score(self):
        spec = {"type": "score", "levels": ["a", "b"]}
        assert is_correct(spec, 1, 1)
        assert not is_correct(spec, 0, 1)

    def test_noul(self):
        spec = {"type": "noul"}
        assert is_correct(spec, True, True)
        assert not is_correct(spec, False, True)

    def test_none_is_incorrect(self):
        spec = {"type": "choice", "options": ["a", "b"]}
        assert not is_correct(spec, None, "a")


class TestCheckpoint:
    def test_save_load_roundtrip(self):
        model = TypedDecisionHead(question_spec=QUESTION_SPEC, hidden_dim=32, dropout=0.0)
        model.eval()
        x = torch.randn(2, 1024)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "head.pt")
            save_head(model, path, temperatures={"is_urgent": 2.5})
            loaded = load_head(path)
            loaded.eval()
            assert loaded.question_spec == QUESTION_SPEC
            assert loaded.temperatures == {"is_urgent": 2.5}
            with torch.no_grad():
                out_orig = model(x)
                out_loaded = loaded(x)
                for qid in QUESTION_SPEC:
                    assert torch.allclose(out_orig[qid], out_loaded[qid], atol=1e-6), qid

    def test_random_heads_differ(self):
        r1 = create_random_head(QUESTION_SPEC, hidden_dim=32)
        r2 = create_random_head(QUESTION_SPEC, hidden_dim=32)
        x = torch.randn(2, 1024)
        assert not torch.allclose(r1(x)["quality"], r2(x)["quality"])


def test_get_device_returns_valid():
    assert isinstance(get_device(), torch.device)