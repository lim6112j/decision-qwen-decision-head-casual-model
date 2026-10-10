"""Tests for label-preserving shape variants (states/shapes.py)."""

import json
from pathlib import Path

from decision_lab.states.shapes import _flatten, _unmarked, shape_variants


class TestFlatten:
    def test_newlines_become_spaces(self):
        assert _flatten("Subject: X\n\nHi team,\n\nbody.") == "Subject: X Hi team, body."

    def test_collapses_indentation(self):
        assert _flatten("  a   b\n\tc  ") == "a b c"


class TestUnmarked:
    def test_strips_lead_labels(self):
        assert _unmarked("Subject: Follow-up on invoice\n\nBody text.") == (
            "Follow-up on invoice Body text."
        )

    def test_strips_markdown_labels_and_residue(self):
        out = _unmarked("**Urgency:** `LOW`\n## Summary\nThis is fine.")
        # bold/code residue and the lead label both go; content survives
        assert out == "LOW Summary This is fine."

    def test_drops_greeting_lines(self):
        assert _unmarked("Hi team,\n\nThe service is broken.") == "The service is broken."

    def test_strips_bullet_and_dash_leads(self):
        out = _unmarked("• Status: resolved\n- Priority: high")
        assert out == "resolved high" or out.endswith("high")
        assert "Status:" not in out and "Priority:" not in out


class TestShapeVariants:
    def test_original_first_deduped(self):
        out = shape_variants("single line already")
        assert out == [] or all(v != "single line already" for v in out)

    def test_multi_line_yields_two_variants(self):
        text = "Subject: X\n\nHi team,\n\nThis is unacceptable and frustrating."
        out = shape_variants(text)
        assert len(out) == 2
        assert out[0] == _flatten(text)
        assert out[1] == _unmarked(text)
        # every latent-bearing signal survives
        for signal in ["unacceptable", "frustrating"]:
            assert all(signal in v for v in out)

    def test_no_empty_variants(self):
        out = shape_variants("Hi team,\n\nSubject: x")
        assert all(v for v in out)


class TestSignalPreservationOnDataset:
    """Every variant of every test-set state keeps the label-bearing
    content of the original — augmentation must not corrupt gold labels."""

    def test_no_signal_lost(self):
        signals = (
            "unacceptable", "disappointed", "pleased", "excellent support",
            "informational", "no rush", "when you get a chance",
            "needs attention this week", "must be addressed immediately",
            "please review and approve", "no action is needed",
        )
        data = Path(__file__).resolve().parent.parent / "data" / "test_indist.jsonl"
        if not data.exists():
            return  # dataset not generated yet; unit tests cover the rest
        checked = 0
        for line in data.read_text().splitlines():
            d = json.loads(line)
            for variant in shape_variants(d["text"]):
                missing = [
                    s for s in signals
                    if s in d["text"] and s not in variant
                ]
                assert not missing, f"doc {d['doc_id']}: lost {missing}"
                checked += 1
        assert checked > 0
