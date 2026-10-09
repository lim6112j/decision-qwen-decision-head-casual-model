"""Tests for the synthetic text-state generator and dataset I/O."""

import tempfile
from dataclasses import replace
from pathlib import Path
from random import Random

from decision_lab.config import load_config
from decision_lab.states.dataset import load_dataset, save_dataset
from decision_lab.states.generator import (
    IN_DIST_TEMPLATES,
    _make_state,
    generate_dataset,
)

BANK_KEYS = {"sentiment", "urgency", "quality", "is_actionable", "contains_pii", "is_urgent"}


def _small_cfg():
    cfg = load_config("configs/default.yaml")
    return replace(cfg,
                   generator=replace(cfg.generator,
                                     num_train=40, num_test_indist=12, num_test_heldout=12,
                                     num_breakout_train=24, num_breakout_test=12))


class TestGenerator:
    def test_labels_exact_and_complete(self):
        cfg = _small_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            generate_dataset(cfg, Path(tmp))
            states = load_dataset(Path(tmp) / "train.jsonl")
            assert len(states) == 40
            for s in states:
                assert set(s.labels.keys()) == BANK_KEYS
                assert s.labels["sentiment"] in ("positive", "negative", "neutral")
                assert s.labels["urgency"] in ("low", "medium", "high", "critical")
                assert 0 <= s.labels["quality"] <= 3
                assert isinstance(s.labels["is_actionable"], bool)
                assert isinstance(s.labels["contains_pii"], bool)
                # derived label: is_urgent iff urgency is high/critical
                assert s.labels["is_urgent"] == (s.labels["urgency"] in ("high", "critical"))

    def test_deterministic_given_seed(self):
        cfg = _small_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            generate_dataset(cfg, Path(tmp) / "a")
            generate_dataset(cfg, Path(tmp) / "b")
            a = (Path(tmp) / "a" / "train.jsonl").read_text()
            b = (Path(tmp) / "b" / "train.jsonl").read_text()
            assert a == b

    def test_heldout_uses_unseen_templates(self):
        cfg = _small_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            generate_dataset(cfg, Path(tmp))
            train_types = {s.state_type for s in load_dataset(Path(tmp) / "train.jsonl")}
            heldout_types = {s.state_type for s in load_dataset(Path(tmp) / "test_heldout.jsonl")}
            assert train_types == set(IN_DIST_TEMPLATES)
            assert heldout_types.isdisjoint(train_types)
            for s in load_dataset(Path(tmp) / "test_heldout.jsonl"):
                assert set(s.labels.keys()) == BANK_KEYS

    def test_pii_signal_in_text(self):
        cfg = _small_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            generate_dataset(cfg, Path(tmp))
            states = load_dataset(Path(tmp) / "train.jsonl")
            with_pii = [s for s in states if s.labels["contains_pii"]]
            without = [s for s in states if not s.labels["contains_pii"]]
            assert with_pii and without
            # PII states mention an email address; redacted ones do not
            assert all("@" in s.text for s in with_pii)
            assert not any("@example.com" in s.text for s in without)


class TestDatasetIO:
    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roundtrip.jsonl"
            states = [_make_state(i, Random(i), IN_DIST_TEMPLATES) for i in range(5)]
            save_dataset(states, path)
            loaded = load_dataset(path)
            assert loaded == states