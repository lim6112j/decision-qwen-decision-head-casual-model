"""Tests for the typed answer parser (prompt LM output)."""

from decision_lab.prompt_lm.parser import parse_typed_answers

QUESTION_SPEC = {
    "sentiment": {
        "type": "choice",
        "options": ["positive", "negative", "neutral"],
        "option_descriptions": {},
    },
    "urgency": {
        "type": "choice",
        "options": ["low", "medium", "high", "critical"],
        "option_descriptions": {},
    },
    "quality": {"type": "score", "levels": ["Poor", "Fair", "Good", "Excellent"]},
    "is_actionable": {"type": "noul", "question": "…"},
}


class TestParseTypedAnswers:
    def test_full_answer(self):
        text = (
            "sentiment: positive\n"
            "urgency: high\n"
            "quality: 2\n"
            "is_actionable: true\n"
        )
        answers = parse_typed_answers(text, QUESTION_SPEC)
        assert answers == {
            "sentiment": "positive",
            "urgency": "high",
            "quality": 2,
            "is_actionable": True,
        }

    def test_case_insensitive_and_casing_normalized(self):
        answers = parse_typed_answers("Sentiment: NEGATIVE\nUrgency: Critical", QUESTION_SPEC)
        assert answers["sentiment"] == "negative"   # normalized to option key casing
        assert answers["urgency"] == "critical"

    def test_bool_variants(self):
        for token, expected in [("true", True), ("yes", True), ("1", True),
                                ("false", False), ("no", False), ("0", False),
                                ("True.", True)]:
            answers = parse_typed_answers(f"is_actionable: {token}", QUESTION_SPEC)
            assert answers["is_actionable"] is expected, token

    def test_score_out_of_range_ignored(self):
        answers = parse_typed_answers("quality: 9", QUESTION_SPEC)
        assert "quality" not in answers

    def test_score_extracts_first_digit(self):
        answers = parse_typed_answers("quality: level 3 of 4", QUESTION_SPEC)
        assert answers["quality"] == 3

    def test_unknown_option_ignored(self):
        answers = parse_typed_answers("sentiment: angry", QUESTION_SPEC)
        assert "sentiment" not in answers

    def test_missing_questions_absent(self):
        answers = parse_typed_answers("sentiment: neutral", QUESTION_SPEC)
        assert answers == {"sentiment": "neutral"}

    def test_cot_output_with_reasoning_line(self):
        text = (
            "Reasoning: The message complains and demands immediate escalation.\n"
            "sentiment: negative\n"
            "urgency: critical\n"
            "quality: 1\n"
            "is_actionable: false\n"
        )
        answers = parse_typed_answers(text, QUESTION_SPEC)
        assert answers["sentiment"] == "negative"
        assert answers["urgency"] == "critical"
        assert answers["is_actionable"] is False

    def test_empty_text(self):
        assert parse_typed_answers("", QUESTION_SPEC) == {}
        assert parse_typed_answers("Blah blah no answers", QUESTION_SPEC) == {}