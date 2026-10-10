"""Compositional question bank for the v4 dynamic head.

Replaces the fixed 5-qid bank + 3-4 handwritten phrasings (the v3 training
signal's bottleneck: FiLM was regulated on ~25 question texts and free in
every other direction). The bank is built deterministically — same seed,
byte-identical bank — and every gold label is read from ``state.labels``
or derived from it by a pure function, so no manual labeling exists.

Structure per qid:
- polarity inversions (NOT-form, gold = exact negation),
- derived predicates (is_urgent-style compositions),
- same-gold/different-text and opposite-gold pairs (the head must route on
  question text, not option vocabulary),
- NEW question types over existing latents (choice/score/noul variety).

Phrasings: the backbone pools the LAST token ("last"), so phrasings are
question-FINAL — prefix modifiers only ("For this document: ..."), never
suffixes; the content word stays last. Cores are hand-written; prefixes
compose deterministically at bank build (frozen tuples, not runtime
generation — the v3 comment about no generated phrasings was about
nondeterminism, which seeded composition avoids). Entry 0 of every qid is
the canonical string used by serving/benchmark, byte-identical to the
v3-era canonical text where one existed.

Domain split: doc entries are applicable to generator.py states only,
breakout entries to states/breakout.py states — ``applicable`` keys off
label availability so a doc question never asks a breakout state.
"""

from dataclasses import dataclass
from itertools import product

from decision_lab.states.breakout import BREAKOUT_STATE_TYPE, PADDLE_QID, MOTION_QID
from decision_lab.states.dataset import TextState

BREAKOUT_OPTIONS = ("left", "right", "stay")

# Deterministic prefix set composed with each core. Question-final pooling:
# every string below keeps the core's content word last.
PREFIXES = ("", "For this document: ", "Regarding the text above: ",
            "Answer about this item: ", "Looking at the passage: ")

# Canonical phrasings (entry 0) are byte-identical to the v3-era
# QUESTION_PHRASINGS entry 0 so serving/benchmark parity holds.
DOC_CORES: dict[str, list[str]] = {
    "sentiment": [
        "What is the sentiment of this message?",
        "sentiment of the message?",
        "How does this message feel tonally?",
        "Is the writer pleased, unhappy, or neutral here?",
        "Which mood does the author convey?",
        "Tone check: positive, negative, or neutral?",
        "Does the text read approvingly or disapprovingly?",
        "What emotional valence does this carry?",
        "Classify the writer's attitude.",
        "Sentiment verdict?",
    ],
    "urgency": [
        "How urgent is this item?",
        "urgency level?",
        "How soon does this need attention?",
        "Which priority tier applies?",
        "Should this be handled immediately or can it wait?",
        "Rate the time pressure.",
        "Is this routine or pressing?",
        "Urgency classification?",
        "How quickly must someone act?",
        "Priority assessment?",
    ],
    "quality": [
        "What is the quality of this text?",
        "quality rating?",
        "How well written is this?",
        "How polished is the writing?",
        "Grade the composition quality.",
        "Is this draft-level or finished writing?",
        "Quality of the document's prose?",
        "How complete does this text look?",
        "Writing standard assessment?",
        "Rate the craftsmanship of this text.",
    ],
    "is_actionable": [
        "Does this state require a response or action?",
        "action needed?",
        "Should the recipient do something about this?",
        "Does this ask for a follow-up?",
        "Is there a call to action here?",
        "Must anyone respond to this?",
        "Action-required check.",
        "Does this demand action or merely inform?",
        "Is this a request or an FYI?",
        "Requires action?",
    ],
    "contains_pii": [
        "Does this state contain personally identifiable information?",
        "any PII present?",
        "Are there personal identifiers in the text?",
        "Does this expose private contact details?",
        "Is sensitive personal data visible?",
        "PII check: names, emails, or phones present?",
        "Could this identify a specific person?",
        "Any private information disclosed?",
        "Personal data exposure?",
        "Identifiable details included?",
    ],
    "is_urgent": [
        "Is this state time-sensitive?",
        "time-sensitive?",
        "Does this need prompt handling?",
        "Is immediate attention warranted?",
        "Should this be escalated now?",
        "Does this belong in the urgent queue?",
        "Time-critical or deferrable?",
        "Is this a now-item?",
        "Needs immediate handling?",
        "Escalation-worthy?",
    ],
    # Polarity inversions — gold is the exact negation of a base predicate.
    "not_urgent": [
        "Is this state NOT time-sensitive?",
        "not time-sensitive?",
        "Can this safely wait?",
        "Is deferring this harmless?",
        "Does this need no prompt handling?",
        "Is this safe to schedule later?",
        "No immediate attention required?",
        "Is this a later-item?",
    ],
    "not_actionable": [
        "Is this state informational only?",
        "informational only?",
        "Does this require no action?",
        "Is no response needed here?",
        "Is this just an FYI?",
        "Can this be archived without follow-up?",
        "No call to action present?",
        "Purely informational?",
    ],
    "no_pii": [
        "Is this state free of personally identifiable information?",
        "PII-free?",
        "Are there no personal identifiers here?",
        "Does this contain no private contact details?",
        "Is sensitive personal data absent?",
        "Nothing personally identifying present?",
        "Is the personal data fully redacted?",
        "No private information disclosed?",
    ],
    "not_negative": [
        "Is the sentiment NOT negative?",
        "not negative?",
        "Is the tone positive or neutral rather than negative?",
        "Does the text avoid a disapproving stance?",
        "Is the writer not unhappy here?",
        "Non-negative sentiment check.",
        "Is the mood anything but negative?",
        "Does the message avoid negativity?",
    ],
    # New question types over existing latents.
    "pii_presence": [
        "Is personal information present or absent?",
        "PII presence?",
        "Does this state show personal data or not?",
        "What is the personal-data status here?",
        "Personal identifiers: present or absent?",
        "Privacy status of this text?",
        "Is identifiable information visible or absent?",
        "PII state?",
    ],
    "contact_mentioned": [
        "Does the text mention any contact information?",
        "contact info mentioned?",
        "Are names, emails, or phone numbers written here?",
        "Does this text include reachable contact details?",
        "Is there a way to contact someone in this text?",
        "Contact presence check.",
        "Any contact details written down?",
        "Are reachable identities mentioned?",
    ],
}

BREAKOUT_CORES: dict[str, list[str]] = {
    PADDLE_QID: [
        "Which direction should the paddle move?",
        "which direction the paddle move?",
        "paddle direction?",
        "Where should the paddle go next?",
        "Which way to move the paddle?",
        "Paddle control for this state?",
        "How should the paddle respond?",
        "What paddle action fits this situation?",
        "Best paddle move right now?",
        "Paddle steering decision?",
    ],
    MOTION_QID: [
        "Which way is the ball moving horizontally?",
        "ball horizontal motion?",
        "Is the ball drifting left or right?",
        "Horizontal ball velocity direction?",
        "How is the ball traveling sideways?",
        "Ball's lateral movement?",
        "Which horizontal way does the ball travel?",
        "Sideways motion of the ball?",
        "Ball drift direction?",
        "Lateral ball direction?",
    ],
    # Derived breakout predicates — exact from latents, NOT paddle gold.
    "is_ball_left": [
        "Is the ball to the left of the paddle?",
        "ball on the left side?",
        "Is the ball positioned left of the paddle?",
        "Does the ball sit on the left half?",
        "Left-side ball check.",
        "Is the ball's horizontal position leftward?",
        "Ball left of center?",
        "Is the ball on the left?",
    ],
    "ball_moving": [
        "Is the ball moving at all horizontally?",
        "is the ball in motion?",
        "Does the ball have horizontal velocity?",
        "Is the ball's sideways speed nonzero?",
        "Is the ball actually moving sideways?",
        "Ball motion present?",
        "Is the ball stationary or moving?",
        "Does the ball drift at all?",
    ],
    "ball_rising": [
        "Is the ball moving upward?",
        "ball rising?",
        "Does the ball travel upward?",
        "Is the ball heading toward the top?",
        "Is the vertical ball motion upward?",
        "Ball moving up or down?",
        "Is the ball climbing?",
        "Upward ball direction?",
    ],
}


@dataclass(frozen=True)
class QuestionBankEntry:
    """One compositional question: text variants + programmatic gold."""

    qid: str
    kind: str                                   # "choice" | "score" | "noul"
    options: tuple[str, ...] | None             # choice/score option texts
    phrasings: tuple[str, ...]                  # entry 0 = canonical
    applicable: "callable[[TextState], bool]"   # domain gate
    gold: "callable[[TextState], object]"       # exact by construction

    def config(self, question_text: str) -> dict:
        """Dynamic question config for one phrasing (serving format)."""
        if self.kind == "choice":
            return {"type": "choice", "options": list(self.options),
                    "question": question_text}
        if self.kind == "score":
            return {"type": "score", "levels": list(self.options),
                    "question": question_text}
        if self.kind == "noul":
            return {"type": "noul", "question": question_text}
        raise ValueError(f"unknown question kind '{self.kind}'")


def _is_doc(state: TextState) -> bool:
    return not state.state_type.startswith(BREAKOUT_STATE_TYPE)


def _is_breakout(state: TextState) -> bool:
    return state.state_type.startswith(BREAKOUT_STATE_TYPE)


def _noul_options() -> tuple[str, ...]:
    return ("false", "true")


def _compose_phrasings(cores: list[str], prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """cores × prefixes, canonical core first (bare, no prefix).

    Deterministic enumeration: prefix-major order after entry 0. Prefix-only
    modification keeps the question-final content word last for last-token
    pooling.
    """
    first = cores[0]
    rest = [
        f"{prefix}{core}"
        for core in cores
        for prefix in prefixes
        if not (core is cores[0] and prefix == "")
    ]
    return (first, *rest)


def _doc_labels(state: TextState) -> dict:
    return state.labels


def _gold_is_ball_left(state: TextState) -> bool:
    """Own label if present; else derive from the paddle gold
    (GOLD_BY_SIDE is 1:1: paddle "left" ⇔ ball on the left)."""
    if "is_ball_left" in state.labels:
        return bool(state.labels["is_ball_left"])
    return state.labels[PADDLE_QID] == "left"


def _gold_ball_moving(state: TextState) -> bool:
    """Own label if present; else derive from the ball gold
    (ball_motion "stay" ⇔ vx == 0)."""
    if "ball_moving" in state.labels:
        return bool(state.labels["ball_moving"])
    return state.labels[MOTION_QID] != "stay"


def build_question_bank() -> tuple[QuestionBankEntry, ...]:
    """Build the full compositional bank (deterministic, no randomness)."""
    entries: list[QuestionBankEntry] = []

    # Doc-domain entries. Golds read state.labels (exact by construction)
    # or derive from them with pure functions.
    doc_specs: list[tuple[str, str, tuple[str, ...] | None, "callable"]] = [
        ("sentiment", "choice", ("positive", "negative", "neutral"),
         lambda s: _doc_labels(s)["sentiment"]),
        ("urgency", "choice", ("low", "medium", "high", "critical"),
         lambda s: _doc_labels(s)["urgency"]),
        ("quality", "score", ("Poor", "Fair", "Good", "Excellent"),
         lambda s: _doc_labels(s)["quality"]),
        ("is_actionable", "noul", None, lambda s: bool(_doc_labels(s)["is_actionable"])),
        ("contains_pii", "noul", None, lambda s: bool(_doc_labels(s)["contains_pii"])),
        ("is_urgent", "noul", None, lambda s: bool(_doc_labels(s)["is_urgent"])),
        # Polarity inversions — exact negations of base predicates
        ("not_urgent", "noul", None, lambda s: not bool(_doc_labels(s)["is_urgent"])),
        ("not_actionable", "noul", None, lambda s: not bool(_doc_labels(s)["is_actionable"])),
        ("no_pii", "noul", None, lambda s: not bool(_doc_labels(s)["contains_pii"])),
        ("not_negative", "noul", None,
         lambda s: _doc_labels(s)["sentiment"] != "negative"),
        # New question types over existing latents
        ("pii_presence", "choice", ("present", "absent"),
         lambda s: "present" if bool(_doc_labels(s)["contains_pii"]) else "absent"),
        ("contact_mentioned", "noul", None, lambda s: bool(_doc_labels(s)["contains_pii"])),
    ]
    for qid, kind, options, gold in doc_specs:
        entries.append(QuestionBankEntry(
            qid=qid,
            kind=kind,
            options=options if kind != "noul" else _noul_options(),
            phrasings=_compose_phrasings(DOC_CORES[qid], PREFIXES),
            applicable=_is_doc,
            gold=gold,
        ))

    # Breakout-domain entries. paddle/ball golds read state.labels; derived
    # predicates read their own label when present (extended
    # BreakoutLatents.labels) and fall back to deriving from the base
    # paddle/ball golds (older datasets predate the extended labels —
    # is_ball_left ⇔ paddle gold "left"; ball_moving ⇔ ball gold ≠ "stay").
    # ball_rising has no base-gold derivation (vy isn't recoverable), so its
    # applicability requires the label key.
    breakout_specs: list[tuple[str, str, tuple[str, ...] | None, "callable"]] = [
        (PADDLE_QID, "choice", BREAKOUT_OPTIONS, lambda s: s.labels[PADDLE_QID]),
        (MOTION_QID, "choice", BREAKOUT_OPTIONS, lambda s: s.labels[MOTION_QID]),
        ("is_ball_left", "noul", None, _gold_is_ball_left),
        ("ball_moving", "noul", None, _gold_ball_moving),
        ("ball_rising", "noul", None, lambda s: bool(s.labels["ball_rising"])),
    ]
    for qid, kind, options, gold in breakout_specs:
        # ball_rising derives from vy, which is NOT recoverable from the
        # base golds — applicability requires the label key (older datasets
        # that predate the extended BreakoutLatents.labels skip it).
        needs_key = qid == "ball_rising"
        def applicable(s, _needs=needs_key):
            return _is_breakout(s) and (not _needs or "ball_rising" in s.labels)
        entries.append(QuestionBankEntry(
            qid=qid,
            kind=kind,
            options=options if kind != "noul" else _noul_options(),
            phrasings=_compose_phrasings(BREAKOUT_CORES[qid], PREFIXES),
            applicable=applicable,
            gold=gold,
        ))

    return tuple(entries)


def canonical_entries(bank: tuple[QuestionBankEntry, ...]) -> dict[str, dict]:
    """Serving-format spec for the bank's canonical phrasings.

    Shape-compatible with build_question_spec output: {qid: config-dict}
    with question text attached.
    """
    spec: dict[str, dict] = {}
    for entry in bank:
        spec[entry.qid] = entry.config(entry.phrasings[0])
    return spec


def phrasing_text(entry: QuestionBankEntry, index: int) -> str:
    """Question text for phrasing index (0 = canonical)."""
    return entry.phrasings[index]
