#!/usr/bin/env python
"""Build ``data_bge/real/labels.jsonl`` for the new ``toxic`` noul question.

The synthetic generator emits no ``toxic`` label, so the dynamic head learns
this question entirely from the real-traffic pipeline. This script authors a
curated seed corpus of short, chat-post-like states (the target distribution),
labels each toxic / not-toxic, and emits one row per (state, question phrasing)
so the head learns the question *region*, not a single string. Both the
``toxic`` and the ``not_toxic`` bank qids are fed (same state, inverted gold) —
the polarity pair that teaches routing on question text.

Run, then build + train::

    python scripts/build_toxic_labels.py            # writes the labels file
    no_proxy=localhost,127.0.0.1 uv run python -m decision_lab \\
        --config configs/bge_full.yaml --data-dir data_bge build-real
    no_proxy=localhost,127.0.0.1 uv run python -m decision_lab \\
        --config configs/bge_full.yaml --data-dir data_bge train-dynamic

Deterministic: ``item_id`` comes from ``real.labels.make_item_id``, so re-runs
are idempotent and dedupe behaves.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decision_lab.real.labels import make_item_id  # noqa: E402

OUT_PATH = pathlib.Path("data_bge/real/labels.jsonl")

# --- Curated corpus --------------------------------------------------------
# Clearly toxic: second-person insults, harassment, demeaning language,
# threats of violence, self-harm encouragement. Deliberately short and
# post-like — this is the distribution the head must generalize over.
TOXIC: tuple[str, ...] = (
    "You are absolutely worthless and everyone here hates you. Do everyone a favor and delete your account, you pathetic loser.",
    "Nobody wants you here. Shut up and get lost, you useless idiot.",
    "You're a complete moron and your opinion means nothing. Kill yourself.",
    "Get out of this community, you disgusting piece of trash.",
    "I hope you die in a fire, you sick freak.",
    "You're so stupid it's a miracle you can even type. Everyone laughs at you.",
    "Shut your mouth, you fat ugly waste of space.",
    "You should be ashamed to exist. The world would be better without you.",
    "What a pathetic excuse for a human being. Go crawl back under your rock.",
    "You're a liar and a thief and everyone knows you're scum.",
    "Stop posting your garbage, no one cares what a loser like you thinks.",
    "You worthless piece of garbage, I'll make sure you regret crossing me.",
    "You're a mentally ill freak and you need to be locked away.",
    "Nobody loves you and nobody ever will, you sad little troll.",
    "You absolute clown, you're too dumb to understand anything anyone says.",
    "Go hang yourself, you don't deserve to breathe.",
    "You're a blight on this platform and we'd all celebrate if you left.",
    "Shut up, you disgusting pig. Your voice makes me sick.",
    "Everyone here is laughing at how pathetic you are.",
    "You're a worthless loser who will never amount to anything.",
    "I'm going to find you and make you pay for this, you trash.",
    "You're the most insufferable, obnoxious imbecile I have ever met.",
    "Die, you pathetic scumbag.",
    "You're ugly, stupid, and completely useless. Do us all a favor and disappear.",
    "You don't belong anywhere near decent people, you creep.",
    "Why don't you just end it? Nobody would miss a loser like you.",
    "You're a joke and a failure and everyone pities you.",
    "Get a life, you miserable loser, and stop bothering normal people.",
    "You are a cancer on this community and should be removed.",
    "Nobody asked, nobody cares, just shut up you idiot.",
    "You're such a worthless human that even your family must be disappointed.",
    "What kind of sick, twisted person writes something this disgusting?",
    "You're a spineless coward and a complete fraud, and I'll expose you.",
    "Crawl back into whatever hole you came from, you filthy animal.",
    "You're not even human, you're just a mistake that should never have been born.",
    "Shut up you dumb foreigner, go back where you came from.",
    "You're a waste of oxygen and everyone would be happier if you vanished.",
    "I'd spit on you if I ever saw your pathetic face in person.",
    "You're the dumbest person on this entire site and that's really saying something.",
    "No one will ever love a repulsive freak like you.",
    "You should be banned forever, you toxic piece of human garbage.",
    "Keep crying, you pathetic baby. Nobody feels sorry for you.",
    "You're a loser with no friends, no life, and no future.",
    "Honestly, the world would be a better place if you just dropped dead.",
    "You disgust me. Everything about you is revolting and wrong.",
    "Get lost, you worthless parasite. Nobody here wants you.",
    "You're a clown, a fraud, and a total embarrassment to everyone around you.",
    "Shut up before I make you shut up, you pathetic little worm.",
    "You contribute nothing but misery. Do everyone a favor and leave.",
    "You're so repulsive that I genuinely can't stand to read a single word you write.",
    "What a sad, lonely, pathetic life you must lead to post something like this.",
    "You're a bully and a coward and I hope life treats you as badly as you treat others.",
    "Nobody respects you, nobody trusts you, and nobody would care if you were gone.",
    "You are a useless, hateful troll and this whole community is sick of you.",
    "Keep talking, you moron, everyone is laughing at how dumb you are.",
)

# Benign: normal messages, requests, thanks, neutral updates, mild logistics.
BENIGN: tuple[str, ...] = (
    "Thanks for the quick turnaround on the invoice. Everything looks correct and we'll process payment on Friday.",
    "Hi team, just a reminder that the standup moves to 10am tomorrow. See you there.",
    "Could you send over the latest version of the deck when you get a chance? No rush.",
    "I've closed ticket #4412 — the issue was a stale cache and the fix is deployed.",
    "Great work on the launch everyone, the metrics look solid so far.",
    "Quick question: which environment should I use for the staging tests?",
    "The meeting notes are in the shared drive under Q3/planning.",
    "Let me know if you need anything else from me before the deadline.",
    "Happy to help with the review — I'll take a look this afternoon.",
    "Confirmed: the shipment arrives Thursday and we'll unload it in the morning.",
    "Please find the attached report. Let me know if any figures need clarification.",
    "I'm out of office until Monday; reach Sam for anything urgent.",
    "The new onboarding docs are live — feedback welcome.",
    "Can we reschedule our 1:1 to Wednesday at 2pm?",
    "Thanks for flagging this; I've logged it with the platform team.",
    "The build passed on CI, so we're clear to merge.",
    "Here's the summary of yesterday's customer call, action items at the bottom.",
    "No worries about the delay — thanks for keeping me posted.",
    "I'll pick up the catering for the offsite; any dietary restrictions?",
    "The dashboard is back up; the outage lasted about twelve minutes.",
    "Reminder to submit your expense reports by end of month.",
    "We're reviewing the proposal and will get back to you early next week.",
    "Nice catch on that bug, that would have been painful in production.",
    "The conference talk went well — about forty people attended.",
    "Let's sync tomorrow to align on the roadmap priorities.",
    "I've shared the folder with the whole team; permissions should be set now.",
    "Appreciate the patience while we sorted out the billing mix-up.",
    "The client approved the design, so we can start implementation.",
    "Here are the notes from the retro; shout if I missed anything.",
    "Booking the room for Thursday — does 3pm still work for everyone?",
    "The new hire starts Monday; I'll set up their accounts before then.",
    "Just merged the fix — please pull and verify on your side.",
    "Thanks again for covering my shift last week, I owe you one.",
    "The report is due Friday, but I can have a draft ready by Wednesday.",
    "Could you approve the pull request when you have a moment?",
    "Everything reconciled — the numbers match the ledger now.",
    "Welcome aboard! Let me know if you need help getting set up.",
    "The server maintenance is scheduled for Saturday night, expect brief downtime.",
)

# Borderline: negative, harsh, frustrated, or profane — but not targeted abuse.
# Labeled NOT toxic; these are the hard negatives that keep the head honest.
BORDERLINE: tuple[str, ...] = (
    "This is really frustrating — the same bug has come back for the third time.",
    "I'm disappointed with the delay, we had agreed on a firm deadline.",
    "Honestly your proposal has some serious problems that we need to discuss.",
    "That's a terrible idea and I think it would hurt the project.",
    "I strongly disagree with this decision and I want it reconsidered.",
    "This process is a mess and it's wasting everyone's time.",
    "The code quality here is poor and it's slowing the whole team down.",
    "I'm annoyed that this wasn't flagged earlier.",
    "Your argument doesn't hold up — you're ignoring the obvious counterexamples.",
    "Wow, that was a genuinely bad call.",
    "I've raised this issue three times now and nothing has changed. Unacceptable.",
    "This draft is weak and I don't think it's ready to ship.",
    "The response time has been unacceptable lately and customers are noticing.",
    "I think you're wrong about this, and here's why.",
    "That meeting could easily have been an email.",
    "The last release caused real problems for users — we need a postmortem.",
    "I'm tired of repeating myself on this. Please read the earlier thread.",
    "This is the worst onboarding experience I've had in years.",
)

# Question phrasings — mirrors DOC_CORES in question_bank.py so the trained
# question region lines up with the bank's canonical/served phrasings.
TOXIC_QUESTIONS: tuple[str, ...] = (
    "Is this text toxic?",
    "toxic?",
    "Does this message contain harassment or abuse?",
    "Is this post hateful or abusive?",
    "Is the language here toxic or harmful?",
    "Does this text attack or demean someone?",
    "Is this content abusive toward a person?",
)

NOT_TOXIC_QUESTIONS: tuple[str, ...] = (
    "Is this text NOT toxic?",
    "not toxic?",
    "Is this message free of harassment or abuse?",
    "Is this message non-toxic?",
    "Does this text avoid abusive language?",
)


def _row(text: str, question: str, is_toxic: bool, idx: int) -> dict:
    """One LabelItem-shaped row; gold_idx 1=True, 0=False (noul order)."""
    return {
        "item_id": make_item_id(text, None, question, ["false", "true"], "noul"),
        "call_id": f"toxic-seed-{idx}",
        "text": text,
        "question": question,
        "kind": "noul",
        "options": ["false", "true"],
        "gold_idx": 1 if is_toxic else 0,
        "source": "accepted",
        "labeled_by": "human:curated-seed",
    }


def build_rows() -> list[dict]:
    """One row per (state, phrasing) for both the toxic and not_toxic qids."""
    states = (
        [(t, True) for t in TOXIC]
        + [(t, False) for t in BENIGN]
        + [(t, False) for t in BORDERLINE]
    )
    rows: list[dict] = []
    for idx, (text, is_toxic) in enumerate(states):
        for q in TOXIC_QUESTIONS:
            rows.append(_row(text, q, is_toxic, idx))
        for q in NOT_TOXIC_QUESTIONS:
            rows.append(_row(text, q, not is_toxic, idx))  # polarity inversion
    return rows


def main() -> None:
    rows = build_rows()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )
    n_states = len(TOXIC) + len(BENIGN) + len(BORDERLINE)
    print(f"Wrote {len(rows)} rows ({n_states} states: "
          f"{len(TOXIC)} toxic, {len(BENIGN)} benign, {len(BORDERLINE)} borderline) "
          f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
