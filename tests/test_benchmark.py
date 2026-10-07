"""Tests for benchmark metrics computation."""

import pytest

from decision_lab.eval.benchmark import EvalRow, compute_metrics


class TestComputeMetrics:
    def test_perfect_accuracy(self):
        rows = [
            EvalRow(agent_id="test", action_id=0, label=0, latency_ms=10.0),
            EvalRow(agent_id="test", action_id=1, label=1, latency_ms=12.0),
            EvalRow(agent_id="test", action_id=2, label=2, latency_ms=8.0),
        ]
        m = compute_metrics(rows)
        assert m.accuracy == 1.0
        assert m.num_parse_failures == 0

    def test_partial_accuracy(self):
        rows = [
            EvalRow(agent_id="test", action_id=0, label=0, latency_ms=10.0),
            EvalRow(agent_id="test", action_id=1, label=2, latency_ms=10.0),  # wrong
            EvalRow(agent_id="test", action_id=None, label=1, latency_ms=10.0),  # parse fail
        ]
        m = compute_metrics(rows)
        assert m.accuracy == 1 / 3
        assert m.num_parse_failures == 1

    def test_latency_stats(self):
        rows = [
            EvalRow(agent_id="test", action_id=0, label=0, latency_ms=10.0),
            EvalRow(agent_id="test", action_id=0, label=0, latency_ms=20.0),
            EvalRow(agent_id="test", action_id=0, label=0, latency_ms=30.0),
            EvalRow(agent_id="test", action_id=0, label=0, latency_ms=40.0),
        ]
        m = compute_metrics(rows)
        assert m.mean_latency_ms == 25.0
        assert m.median_latency_ms == 25.0

    def test_empty_rows(self):
        rows = []
        m = compute_metrics(rows)
        assert m.accuracy == 0.0
        assert m.num_parse_failures == 0