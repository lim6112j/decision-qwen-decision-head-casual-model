"""Tests for field splitting (states/fields.py).

Covers: strategy selection (JSON / newline / log token / sentence),
single-field fallback, MAX_FIELDS cap, summary field, and — critically —
chunking consistency between the heuristic splitter and the breakout
renderer fields (training-time chunking must equal inference-time chunking).
"""

import pytest

from decision_lab.states.dataset import TextState
from decision_lab.states.fields import (
    MAX_FIELDS,
    split_sentences,
    split_state_fields,
    state_field_set,
    state_fields,
)


class TestSplitStateFields:
    def test_json_object_flattens_to_leaves(self):
        text = '{"ball": {"x": 369, "vx": 2}, "paddle": {"x": 382}, "bricks": 39}'
        fields = split_state_fields(text)
        assert fields == ["ball.x: 369", "ball.vx: 2", "paddle.x: 382", "bricks: 39"]

    def test_json_arrays_join(self):
        text = '{"items": ["a", "b"], "n": 2}'
        fields = split_state_fields(text)
        assert fields == ["items: a, b", "n: 2"]

    def test_newline_split(self):
        text = "first line\nsecond line\n\nthird line"
        assert split_state_fields(text) == ["first line", "second line", "third line"]

    def test_log_tokens_split(self):
        text = "GAP=+124 VX=+2 VY=-1 BRICKS=17 SCORE=88 LIVES=2"
        assert split_state_fields(text) == [
            "GAP=+124", "VX=+2", "VY=-1", "BRICKS=17", "SCORE=88", "LIVES=2",
        ]

    def test_sentences_split(self):
        text = "First sentence here. Second one! Third?"
        assert split_state_fields(text) == [
            "First sentence here.", "Second one!", "Third?",
        ]

    def test_plain_text_falls_back_to_single_field(self):
        assert split_state_fields("no delimiters at all") == ["no delimiters at all"]

    def test_empty_text_yields_single_empty_field(self):
        assert split_state_fields("   ") == [""]

    def test_malformed_json_falls_through(self):
        text = '{"unterminated": [1, 2'
        fields = split_state_fields(text)
        assert len(fields) >= 1  # no crash; falls to another strategy

    def test_cap_merges_to_max_fields(self):
        text = "\n".join(f"line {i}" for i in range(40))
        fields = split_state_fields(text)
        assert len(fields) <= MAX_FIELDS
        assert len(fields) >= 2
        assert "line 0" in fields[0]          # content preserved, not dropped
        assert "line 39" in fields[-1]

    def test_sentences_with_multi_sentence_status(self):
        """A field containing several sentences splits per sentence."""
        text = ("The ball is clearly to the LEFT of the paddle (gap 178 px) "
                "and moving right and up, away from the paddle. "
                "Bricks remaining: 29. Score: 189. Lives: 3.")
        fields = split_state_fields(text)
        assert fields == [
            "The ball is clearly to the LEFT of the paddle (gap 178 px) "
            "and moving right and up, away from the paddle.",
            "Bricks remaining: 29.",
            "Score: 189.",
            "Lives: 3.",
        ]


class TestSplitSentences:
    def test_basic(self):
        assert split_sentences("A. B.") == ["A.", "B."]

    def test_no_split_on_comma(self):
        assert split_sentences("one, two, three.") == ["one, two, three."]

    def test_single_no_terminal(self):
        assert split_sentences("just words") == ["just words"]


class TestStateFields:
    def test_explicit_fields_win(self):
        state = TextState(doc_id=0, state_type="t", text="a. b.", labels={},
                          fields=["explicit"])
        assert state_fields(state) == ["explicit"]

    def test_none_fields_use_heuristic(self):
        state = TextState(doc_id=0, state_type="t", text="a. b.", labels={})
        assert state_fields(state) == ["a.", "b."]

    def test_no_fields_means_full_text(self):
        state = TextState(doc_id=0, state_type="t", text="single chunk", labels={})
        assert state_fields(state) == ["single chunk"]


class TestStateFieldSet:
    def test_summary_prepended(self):
        state = TextState(doc_id=0, state_type="t", text="whole text", labels={},
                          fields=["f1", "f2"])
        assert state_field_set(state, include_summary=True) == [
            "whole text", "f1", "f2",
        ]

    def test_no_summary(self):
        state = TextState(doc_id=0, state_type="t", text="whole text", labels={},
                          fields=["f1", "f2"])
        assert state_field_set(state, include_summary=False) == ["f1", "f2"]

    def test_summary_guarantees_nonempty(self):
        """Even a whitespace-only text yields M >= 1 with a summary field."""
        state = TextState(doc_id=0, state_type="t", text="   ", labels={})
        fields = state_field_set(state, include_summary=True)
        assert len(fields) >= 1 and fields[0] == "   "


class TestDegenerateSetGuard:
    """A field set whose entries are all identical defeats the head:

    identical attention keys → uniform attention → one context vector →
    exactly equal logits for every option. state_field_set must guarantee
    >= 2 distinct entries whenever the text can be split at all.
    """

    def test_single_sentence_with_summary_gets_clause_split(self):
        text = ("The ball is clearly to the LEFT of the paddle (gap 124 px) "
                "and moving right and up, away from the paddle.")
        state = TextState(doc_id=0, state_type="t", text=text, labels={})
        fields = state_field_set(state, include_summary=True)
        # summary + the one sentence + 2 comma clauses
        assert len(fields) == 4
        assert len(set(fields)) >= 2
        assert fields[0] == text
        assert fields[2] == ("The ball is clearly to the LEFT of the paddle "
                             "(gap 124 px) and moving right and up")
        assert fields[3] == "away from the paddle."

    def test_no_delimiters_falls_back_to_word_chunks(self):
        text = "one two three four five six seven eight nine"
        state = TextState(doc_id=0, state_type="t", text=text, labels={})
        fields = state_field_set(state, include_summary=True)
        # summary + text + 2 word chunks (9 words / 6 per chunk)
        assert len(fields) == 4
        assert len(set(fields)) >= 2

    def test_two_word_text_yields_two_distinct_fields(self):
        text = "short state"
        state = TextState(doc_id=0, state_type="t", text=text, labels={})
        fields = state_field_set(state, include_summary=True)
        assert len(fields) >= 3 and len(set(fields)) >= 2

    def test_single_word_stays_degenerate(self):
        """A one-word text cannot be split — [text, text] is the best any
        splitter can do."""
        state = TextState(doc_id=0, state_type="t", text="ready", labels={})
        assert state_field_set(state, include_summary=True) == ["ready", "ready"]

    def test_multi_field_sets_are_untouched(self):
        """States that already resolve to >= 2 distinct fields must come
        back byte-identical — the guard only fires on degenerate sets."""
        state = TextState(doc_id=0, state_type="t",
                          text="first sentence. second sentence.", labels={})
        assert state_field_set(state, include_summary=True) == [
            "first sentence. second sentence.", "first sentence.", "second sentence.",
        ]

    def test_guard_applies_without_summary(self):
        state = TextState(doc_id=0, state_type="t",
                          text="clause one, clause two.", labels={})
        assert state_field_set(state, include_summary=False) == [
            "clause one, clause two.", "clause one", "clause two.",
        ]


class TestBreakoutChunkingConsistency:
    """The invariant the service depends on: heuristic split of a rendered
    breakout text must equal the renderer's stored field set, because
    HTTP custom_text callers get the heuristic while training stored
    renderer fields."""

    def _iterate_states(self, n: int = 400):
        from random import Random
        from decision_lab.states.breakout import make_breakout_state

        rng = Random(123)  # different seed from generation: covers variety
        for i in range(n):
            yield make_breakout_state(i, rng)

    def test_heuristic_split_matches_renderer_fields(self):
        seen_templates = set()
        for state in self._iterate_states():
            seen_templates.add(state.state_type)
            if state.fields is None:
                continue
            assert split_state_fields(state.text) == state.fields, (
                f"{state.state_type}: renderer fields {state.fields!r} != "
                f"heuristic {split_state_fields(state.text)!r}"
            )
        # All 5 templates must be covered by the sample
        assert len(seen_templates) == 5

    def test_structured_templates_always_multiple_fields(self):
        """structured/log/prose_vague/prose_short always yield real field sets.
        (prose may legitimately be 1 field for geometry-only texts, but its
        equality with the heuristic is asserted above either way.)"""
        for state in self._iterate_states(200):
            if state.state_type == "breakout_prose":
                continue
            assert len(state.fields) >= 2, (
                f"{state.state_type}: only {len(state.fields)} field"
            )
