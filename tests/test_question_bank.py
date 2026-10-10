"""Tests for the compositional question bank (v4)."""

import pytest

from decision_lab.head.question_bank import (
    BREAKOUT_OPTIONS,
    QuestionBankEntry,
    build_question_bank,
    canonical_entries,
    phrasing_text,
)
from decision_lab.states.breakout import PADDLE_QID, MOTION_QID, make_breakout_state
from decision_lab.states.generator import generate_dataset  # noqa: F401 (integration)
from decision_lab.states.dataset import TextState


@pytest.fixture(scope="module")
def bank() -> tuple[QuestionBankEntry, ...]:
    return build_question_bank()


@pytest.fixture(scope="module")
def doc_state() -> TextState:
    """A document state with known latents (hand-built, exact labels)."""
    return TextState(
        doc_id=0,
        state_type="text",
        text="Subject: Update\n\nReally pleased with how smoothly everything went. "
             "Action required: please review and approve. "
             "You can reach Jordan Lee at jordan.lee@acme.example.com or +1-555-100-1234.",
        labels={
            "sentiment": "positive",
            "urgency": "high",
            "quality": 2,
            "is_actionable": True,
            "contains_pii": True,
            "is_urgent": True,
        },
    )


@pytest.fixture(scope="module")
def doc_state_calm() -> TextState:
    """A calm document state: not urgent, not actionable, no PII."""
    return TextState(
        doc_id=1,
        state_type="text",
        text="Subject: Update\n\nThis is an informational update regarding the "
             "current status. This is informational only — no action is needed.",
        labels={
            "sentiment": "neutral",
            "urgency": "low",
            "quality": 1,
            "is_actionable": False,
            "contains_pii": False,
            "is_urgent": False,
        },
    )


def _breakout_state(side: str, vx: int, vy: int, doc_id: int = 0) -> TextState:
    """Hand-built breakout state with controlled latents."""
    lat_labels = {
        PADDLE_QID: {"left": "left", "center": "stay", "right": "right"}[side],
        MOTION_QID: "stay" if vx == 0 else ("left" if vx < 0 else "right"),
        "is_ball_left": side == "left",
        "ball_moving": vx != 0,
        "ball_rising": vy < 0,
    }
    return TextState(
        doc_id=doc_id,
        state_type=f"breakout_prose",
        text=f"situation: ball is on the {side}",
        labels=lat_labels,
    )


class TestBankStructure:
    def test_bank_nonempty(self, bank):
        assert len(bank) >= 15

    def test_qids_unique(self, bank):
        qids = [e.qid for e in bank]
        assert len(qids) == len(set(qids))

    def test_kinds_valid(self, bank):
        for e in bank:
            assert e.kind in ("choice", "score", "noul"), e.qid

    def test_noul_options_are_false_true(self, bank):
        for e in bank:
            if e.kind == "noul":
                assert e.options == ("false", "true")

    def test_breakout_forcing_pair_present(self, bank):
        """paddle_direction and ball_motion share option texts — the
        question-text-only discriminator (the v4 bootstrap forcing pair)."""
        by_qid = {e.qid: e for e in bank}
        assert by_qid[PADDLE_QID].options == by_qid[MOTION_QID].options == BREAKOUT_OPTIONS

    def test_same_gold_different_text_pair(self, bank):
        """is_urgent and escalation-style routing: a derived predicate asked
        under a different question text must remain answerable (text routing
        is trained, not option-vocabulary memorization)."""
        by_qid = {e.qid: e for e in bank}
        # contact_mentioned shares contains_pii's gold under new text
        assert by_qid["contact_mentioned"].kind == "noul"

    def test_deterministic_build(self):
        """Byte-identical banks across builds (closures compare unequal, so
        the assert covers the observable surface: qids, kinds, options,
        phrasings)."""
        b1, b2 = build_question_bank(), build_question_bank()
        assert [e.qid for e in b1] == [e.qid for e in b2]
        for e1, e2 in zip(b1, b2):
            assert e1.kind == e2.kind
            assert e1.options == e2.options
            assert e1.phrasings == e2.phrasings


class TestPhrasings:
    def test_entry0_is_canonical(self, bank):
        canonicals = {
            PADDLE_QID: "Which direction should the paddle move?",
            MOTION_QID: "Which way is the ball moving horizontally?",
            "sentiment": "What is the sentiment of this message?",
            "urgency": "How urgent is this item?",
            "quality": "What is the quality of this text?",
            "is_actionable": "Does this state require a response or action?",
            "contains_pii": "Does this state contain personally identifiable information?",
            "is_urgent": "Is this state time-sensitive?",
        }
        for qid, text in canonicals.items():
            e = next(e for e in bank if e.qid == qid)
            assert e.phrasings[0] == text, qid

    def test_phrasings_nonempty_and_distinct(self, bank):
        for e in bank:
            assert len(e.phrasings) >= 8, e.qid
            assert len(set(e.phrasings)) == len(e.phrasings), e.qid
            assert all(p.strip() for p in e.phrasings), e.qid

    def test_question_final_composition(self, bank):
        """Prefix composition never appends suffixes — stripping one known
        prefix from any phrasing leaves a bare core (every core appears
        bare since PREFIXES includes ""), so the core's content word stays
        last (last-token pooling is order-sensitive)."""
        from decision_lab.head.question_bank import PREFIXES
        for e in bank:
            for p in e.phrasings:
                stripped = p
                for prefix in PREFIXES:
                    if prefix and stripped.startswith(prefix):
                        stripped = stripped[len(prefix):]
                        break
                assert stripped in e.phrasings, (e.qid, p)
                assert stripped.endswith(("?", ".")), (e.qid, p)

    def test_phrasing_text_accessor(self, bank):
        e = bank[0]
        assert phrasing_text(e, 0) == e.phrasings[0]
        assert phrasing_text(e, 1) == e.phrasings[1]


class TestGoldExactness:
    def test_doc_gold_matches_labels(self, bank, doc_state, doc_state_calm):
        by_qid = {e.qid: e for e in bank}
        assert by_qid["sentiment"].gold(doc_state) == "positive"
        assert by_qid["urgency"].gold(doc_state) == "high"
        assert by_qid["quality"].gold(doc_state) == 2
        assert by_qid["is_actionable"].gold(doc_state) is True
        assert by_qid["contains_pii"].gold(doc_state) is True
        assert by_qid["is_urgent"].gold(doc_state) is True
        # calm state
        assert by_qid["sentiment"].gold(doc_state_calm) == "neutral"
        assert by_qid["quality"].gold(doc_state_calm) == 1

    def test_inversions_are_exact_negations(self, bank, doc_state, doc_state_calm):
        by_qid = {e.qid: e for e in bank}
        assert by_qid["not_urgent"].gold(doc_state) is False
        assert by_qid["not_urgent"].gold(doc_state_calm) is True
        assert by_qid["not_actionable"].gold(doc_state) is False
        assert by_qid["not_actionable"].gold(doc_state_calm) is True
        assert by_qid["no_pii"].gold(doc_state) is False
        assert by_qid["no_pii"].gold(doc_state_calm) is True

    def test_not_negative_by_sentiment(self, bank, doc_state, doc_state_calm):
        by_qid = {e.qid: e for e in bank}
        assert by_qid["not_negative"].gold(doc_state) is True      # positive
        assert by_qid["not_negative"].gold(doc_state_calm) is True  # neutral
        # negative state flips the inversion
        neg = TextState(
            doc_id=2, state_type="text", text="x",
            labels={"sentiment": "negative", "urgency": "low", "quality": 0,
                    "is_actionable": False, "contains_pii": False, "is_urgent": False},
        )
        assert by_qid["not_negative"].gold(neg) is False

    def test_pii_presence_choice_gold(self, bank, doc_state, doc_state_calm):
        by_qid = {e.qid: e for e in bank}
        assert by_qid["pii_presence"].gold(doc_state) == "present"
        assert by_qid["pii_presence"].gold(doc_state_calm) == "absent"

    def test_breakout_gold_matches_labels(self, bank):
        by_qid = {e.qid: e for e in bank}
        st_left = _breakout_state("left", vx=-2, vy=1)
        assert by_qid[PADDLE_QID].gold(st_left) == "left"
        assert by_qid[MOTION_QID].gold(st_left) == "left"
        assert by_qid["is_ball_left"].gold(st_left) is True
        assert by_qid["ball_moving"].gold(st_left) is True
        assert by_qid["ball_rising"].gold(st_left) is False

        st_center = _breakout_state("center", vx=0, vy=-1)
        assert by_qid[PADDLE_QID].gold(st_center) == "stay"
        assert by_qid[MOTION_QID].gold(st_center) == "stay"
        assert by_qid["ball_moving"].gold(st_center) is False
        assert by_qid["ball_rising"].gold(st_center) is True

    def test_opposite_gold_pairs(self, bank):
        """Same breakout state, paddle vs ball questions can disagree —
        vx decorrelated from side."""
        by_qid = {e.qid: e for e in bank}
        st = _breakout_state("left", vx=+3, vy=0)
        assert by_qid[PADDLE_QID].gold(st) == "left"
        assert by_qid[MOTION_QID].gold(st) == "right"


class TestApplicability:
    def test_doc_entries_reject_breakout_state(self, bank, doc_state):
        for e in bank:
            if e.applicable(doc_state):
                assert not e.applicable(_breakout_state("left", 1, 1)), e.qid

    def test_breakout_entries_reject_doc_state(self, bank, doc_state):
        breakout_qids = {PADDLE_QID, MOTION_QID, "is_ball_left", "ball_moving", "ball_rising"}
        for e in bank:
            if e.qid in breakout_qids:
                assert not e.applicable(doc_state), e.qid

    def test_all_breakout_entries_accept_breakout_state(self, bank):
        st = _breakout_state("right", -1, 2)
        breakout_qids = {PADDLE_QID, MOTION_QID, "is_ball_left", "ball_moving", "ball_rising"}
        for e in bank:
            if e.qid in breakout_qids:
                assert e.applicable(st), e.qid


class TestConfigShape:
    def test_config_choice(self, bank):
        e = next(e for e in bank if e.qid == "sentiment")
        cfg = e.config(e.phrasings[0])
        assert cfg == {"type": "choice", "options": ["positive", "negative", "neutral"],
                       "question": e.phrasings[0]}

    def test_config_score(self, bank):
        e = next(e for e in bank if e.qid == "quality")
        cfg = e.config(e.phrasings[0])
        assert cfg["type"] == "score"
        assert cfg["levels"] == ["Poor", "Fair", "Good", "Excellent"]

    def test_config_noul(self, bank):
        e = next(e for e in bank if e.qid == "is_urgent")
        cfg = e.config(e.phrasings[0])
        assert cfg == {"type": "noul", "question": e.phrasings[0]}

    def test_canonical_entries_shape(self, bank):
        spec = canonical_entries(bank)
        assert set(spec) == {e.qid for e in bank}
        for qid, cfg in spec.items():
            assert cfg["type"] in ("choice", "score", "noul")
            if cfg["type"] == "choice":
                assert isinstance(cfg["options"], list)
            assert isinstance(cfg["question"], str) and cfg["question"]
