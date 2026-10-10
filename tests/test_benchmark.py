"""Tests for typed-question benchmark metrics (accuracy, ECE, latency)."""

import numpy as np
import pytest
import torch

from decision_lab.config import Config
from decision_lab.eval.benchmark import (
    EvalRow,
    _load_dynamic_for_benchmark,
    compute_metrics,
    expected_calibration_error,
)
from decision_lab.head.dynamic_model import DynamicDecisionHead

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
        assert m.per_label_accuracy == {}
        assert m.confusion == {}

    def test_confusion_and_per_label_accuracy_expose_bias(self):
        """Majority-prediction collapse shows up as zero per-label accuracy."""
        spec = {
            "dir": {"type": "choice", "options": ["left", "right", "stay"]},
        }
        rows = [
            _row({"dir": "right"}, {"dir": "left"}),
            _row({"dir": "right"}, {"dir": "right"}),
            _row({"dir": "right"}, {"dir": "stay"}),
            _row({"dir": "right"}, {"dir": "left"}),
        ]
        m = compute_metrics(rows, spec)
        pla = m.per_label_accuracy["dir"]
        assert pla["left"] == 0.0          # always predicted "right" when gold=left
        assert pla["right"] == 1.0
        assert m.confusion["dir"]["left"] == {"right": 2}
        assert m.confusion["dir"]["right"] == {"right": 1}
        assert m.confusion["dir"]["stay"] == {"right": 1}


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


class TestLoadDynamicForBenchmark:
    """The benchmark loader must fall back only on a *missing* checkpoint,
    and raise loudly on a checkpoint that exists but fails to load (e.g. a
    stale LoRA-tainted artifact from an older architecture)."""

    @pytest.fixture
    def cfg(self):
        return Config()  # dataclass defaults include dynamic_head.* attrs

    def test_missing_checkpoint_falls_back_to_random(self, cfg, tmp_path):
        head, temperature = _load_dynamic_for_benchmark(
            cfg, tmp_path, torch.device("cpu"),
        )
        assert isinstance(head, DynamicDecisionHead)
        assert temperature == 1.0

    def test_corrupt_checkpoint_raises(self, cfg, tmp_path):
        # Simulate the stale LoRA-adapter checkpoint: valid state_dict plus
        # extra keys the current architecture no longer has.
        state = dict(DynamicDecisionHead().state_dict())
        state["adapter.lora_A.weight"] = torch.zeros(64, 1024)
        state["adapter.lora_B.weight"] = torch.zeros(1024, 64)
        torch.save(
            {
                "model_state": state,
                "arch": {"type": "dynamic", "input_dim": 1024, "hidden_dim": 256, "d_k": 128},
                "temperature": 0.7,
            },
            tmp_path / "head_dynamic.pt",
        )
        with pytest.raises(RuntimeError, match="retrain"):
            _load_dynamic_for_benchmark(cfg, tmp_path, torch.device("cpu"))

    def test_valid_checkpoint_loads(self, cfg, tmp_path):
        head = DynamicDecisionHead(state_set=True)   # v4 is state_set
        torch.save(
            {
                "model_state": head.state_dict(),
                "arch": {
                    "type": "dynamic",
                    "state_set": True,
                    "arch_version": 4,
                    "input_dim": head.input_dim,
                    "hidden_dim": head.hidden_dim,
                    "d_k": head.d_k,
                },
                "temperature": 2.5,
            },
            tmp_path / "head_dynamic.pt",
        )
        loaded, temperature = _load_dynamic_for_benchmark(
            cfg, tmp_path, torch.device("cpu"),
        )
        assert isinstance(loaded, DynamicDecisionHead)
        assert temperature == 2.5

class TestBenchmarkDynamicPhrasings:
    """The v4 held-out-phrasing eval must actually run end to end.

    Regression guard: this function shipped with a NameError (is_correct_dynamic
    not imported) that the whole suite passed through, because nothing
    exercised it. It is the row that measures the v4 generalization claim.
    """

    @pytest.fixture
    def cfg(self):
        return Config()

    class _FakeServer:
        """Deterministic embedder: hash-free, index-stable, right shape."""

        def __init__(self, dim=64):
            self.dim = dim

        def embed(self, texts):
            out = []
            for t in texts:
                g = torch.Generator().manual_seed(abs(hash(t)) % (2**31))
                out.append(torch.randn(self.dim, generator=g).numpy())
            return out

    def _write_split(self, data_dir, name, states, dim):
        from decision_lab.backbone.features import extract_field_features  # noqa
        from decision_lab.states.dataset import save_dataset

        save_dataset(states, data_dir / f"{name}.jsonl")
        # hand-build the ragged field cache: 1 field per state, `dim` wide
        rng = np.random.RandomState(0)
        feats = rng.randn(len(states), dim).astype(np.float32)
        counts = np.ones(len(states), dtype=np.int64)
        np.savez_compressed(
            data_dir / f"features_{name}_fields.npz",
            features=feats, field_counts=counts, fingerprint="test",
        )

    def test_runs_and_reports_per_qid_accuracy(self, cfg, tmp_path):
        from decision_lab.eval.benchmark import _benchmark_dynamic_phrasings
        from decision_lab.head.dynamic_model import save_dynamic_head
        from decision_lab.states.dataset import TextState

        dim = 64
        states = [
            TextState(
                doc_id=i, state_type="text", text=f"doc {i}",
                labels={"sentiment": "positive", "urgency": "low",
                        "quality": 1, "is_actionable": False,
                        "contains_pii": False, "is_urgent": False},
            )
            for i in range(3)
        ]
        self._write_split(tmp_path, "test_indist", states, dim)

        head = DynamicDecisionHead(input_dim=dim, hidden_dim=32, d_k=16,
                                   dropout=0.0, state_set=True)
        save_dynamic_head(head, tmp_path / cfg.dynamic_head.checkpoint_filename)

        cfg_bench = Config()
        metrics = _benchmark_dynamic_phrasings(
            tmp_path, cfg_bench, self._FakeServer(dim), tmp_path,
            cfg_bench.benchmark, {},
        )
        assert metrics is not None
        assert "test_indist" in metrics
        per_q = metrics["test_indist"].per_question_accuracy
        # non-canonical phrasings of bank qids are evaluated
        assert "sentiment" in per_q
        assert "not_urgent" in per_q        # qid outside the fixed bank
        assert all(0.0 <= a <= 1.0 for a in per_q.values())

    def test_missing_splits_return_none(self, cfg, tmp_path):
        from decision_lab.eval.benchmark import _benchmark_dynamic_phrasings

        metrics = _benchmark_dynamic_phrasings(
            tmp_path, cfg, self._FakeServer(64), tmp_path, cfg.benchmark, {},
        )
        assert metrics is None
