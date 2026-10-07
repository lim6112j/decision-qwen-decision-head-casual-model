"""Tests for typed-question benchmark metrics (accuracy, ECE, latency)."""

import numpy as np

from decision_lab.eval.benchmark import (
    EvalRow,
    compute_metrics,
    expected_calibration_error,
)

QUESTION_SPEC = {
    "sentiment": {
        "type": "choice",
        "options": ["positive", "negative", "neutral"],
        "option_descriptions": {},
    },
    "quality": {"type": "score", "levels": ["Poor", "Fair", "Good", "Excellent"]},
    "is_urgent": {"type": "noul", "question": "…"},
}


def _row(predictions, gold, confidence=None, latency=10.0):
    return EvalRow(
        agent_id="test",
        doc_id=0,
        predictions=predictions,
        gold_labels=gold,
        confidence=confidence or {},
        latency_ms=latency,
    )


class TestComputeMetrics:
    def test_perfect_accuracy(self):
        rows = [
            _row({"sentiment": "positive", "quality": 2, "is_urgent": True},
                 {"sentiment": "positive", "quality": 2, "is_urgent": True}),
            _row({"sentiment": "negative", "quality": 0, "is_urgent": False},
                 {"sentiment": "negative", "quality": 0, "is_urgent": False}),
        ]
        m = compute_metrics(rows, QUESTION_SPEC)
        assert m.mean_accuracy == 1.0
        assert all(a == 1.0 for a in m.per_question_accuracy.values())
        assert m.num_parse_failures == 0

    def test_partial_accuracy_and_parse_failures(self):
        rows = [
            _row({"sentiment": "positive", "quality": 2},
                 {"sentiment": "positive", "quality": 2, "is_urgent": True}),
            _row({"sentiment": "negative", "quality": 1},
                 {"sentiment": "negative", "quality": 0, "is_urgent": False}),
        ]
        m = compute_metrics(rows, QUESTION_SPEC)
        assert abs(m.per_question_accuracy["sentiment"] - 1.0) < 1e-9
        assert abs(m.per_question_accuracy["quality"] - 0.5) < 1e-9
        assert m.per_question_accuracy["is_urgent"] == 0.0   # both missing
        assert m.num_parse_failures == 2

    def test_case_insensitive_choice_scoring(self):
        rows = [_row({"sentiment": "POSITIVE"}, {"sentiment": "positive"})]
        m = compute_metrics(rows, QUESTION_SPEC)
        assert m.per_question_accuracy["sentiment"] == 1.0

    def test_latency_stats(self):
        rows = [
            _row({}, {}, latency=10.0),
            _row({}, {}, latency=20.0),
            _row({}, {}, latency=30.0),
            _row({}, {}, latency=40.0),
        ]
        m = compute_metrics(rows, QUESTION_SPEC)
        assert m.mean_latency_ms == 25.0
        assert m.median_latency_ms == 25.0

    def test_empty_rows(self):
        m = compute_metrics([], QUESTION_SPEC)
        assert m.mean_accuracy == 0.0
        assert m.ece == 0.0


class TestECE:
    def test_perfectly_calibrated(self):
        # Within each confidence bin, accuracy equals the mean confidence
        confs = np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75])
        corrects = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=float)
        assert expected_calibration_error(confs, corrects) < 1e-9

    def test_overconfident_has_high_ece(self):
        confs = np.array([1.0] * 10)
        corrects = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        ece = expected_calibration_error(confs, corrects)
        assert abs(ece - 0.9) < 1e-9

    def test_empty(self):
        assert expected_calibration_error(np.array([]), np.array([])) == 0.0

    def test_confident_and_correct_is_low_ece(self):
        confs = np.array([0.99] * 10)
        corrects = np.ones(10)
        assert expected_calibration_error(confs, corrects) < 0.05