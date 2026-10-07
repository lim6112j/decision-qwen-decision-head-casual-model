"""Tests for prompt LM action parser."""

import pytest

from decision_lab.prompt_lm.parser import parse_action


class TestParseAction:
    def test_parse_all_actions(self):
        cases = [
            ("Answer: up", 0),
            ("Answer: down", 1),
            ("Answer: left", 2),
            ("Answer: right", 3),
            ("Answer: wait", 4),
        ]
        for text, expected in cases:
            assert parse_action(text) == expected

    def test_parse_case_insensitive(self):
        assert parse_action("answer: UP") == 0
        assert parse_action("ANSWER: Down") == 1

    def test_parse_with_extra_text(self):
        assert parse_action("Reasoning: the goal is below.\nAnswer: down") == 1

    def test_parse_shortcuts(self):
        assert parse_action("Answer: u") == 0
        assert parse_action("Answer: d") == 1
        assert parse_action("Answer: l") == 2
        assert parse_action("Answer: r") == 3
        assert parse_action("Answer: w") == 4

    def test_parse_invalid_returns_none(self):
        assert parse_action("Blah blah") is None
        assert parse_action("") is None
        assert parse_action("Answer: jump") is None  # not a valid action

    def test_parse_no_answer_prefix(self):
        assert parse_action("up") is None