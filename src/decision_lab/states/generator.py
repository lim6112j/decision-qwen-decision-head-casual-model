"""Synthetic text-state generator with gold labels exact by construction.

Every document is rendered from sampled latent variables; the latents ARE the
gold labels, so supervision requires no annotation step. Templates emit all
latent signals (sentiment, urgency, quality, actionability, PII, time
sensitivity) in surface text.
"""

from dataclasses import dataclass
from pathlib import Path
from random import Random

from decision_lab.config import Config
from decision_lab.states.dataset import TextState, save_dataset

IN_DIST_TEMPLATES: tuple[str, ...] = (
    "email", "ticket", "json", "text", "log_entry", "chat_message", "markdown", "bullet_list",
)

SENTIMENTS: tuple[str, ...] = ("positive", "negative", "neutral")
URGENCIES: tuple[str, ...] = ("low", "medium", "high", "critical")
MAX_QUALITY: int = 3  # rubric levels 0..MAX_QUALITY

FIRST_NAMES: tuple[str, ...] = ("Jordan", "Riley", "Casey", "Morgan", "Avery", "Quinn")
LAST_NAMES: tuple[str, ...] = ("Lee", "Patel", "Chen", "Novak", "Garcia", "Okafor")
COMPANIES: tuple[str, ...] = ("Northwind", "Acme", "Globex", "Initech", "Umbrella")


@dataclass
class Latents:
    """Latent variables that drive both rendering and gold labels."""

    sentiment: str
    urgency: str
    quality: int          # 0..3 ordinal document-quality rubric
    actionable: bool      # does the doc request an action?
    pii: bool             # does the doc contain personal identifiers?

    def labels(self) -> dict:
        """Gold labels for the fixed question bank (exact by construction)."""
        return {
            "sentiment": self.sentiment,
            "urgency": self.urgency,
            "quality": self.quality,
            "is_actionable": self.actionable,
            "contains_pii": self.pii,
            "is_urgent": self.urgency in ("high", "critical"),
        }


def generate_dataset(cfg: Config, data_dir: Path) -> None:
    """Generate train / test_indist / test_heldout JSONL datasets."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = Random(cfg.generator.seed)

    splits = [
        ("train", cfg.generator.num_train, IN_DIST_TEMPLATES),
        ("test_indist", cfg.generator.num_test_indist, IN_DIST_TEMPLATES),
        ("test_heldout", cfg.generator.num_test_heldout, tuple(cfg.generator.heldout_templates)),
    ]

    doc_id = 0
    for name, count, templates in splits:
        states = []
        for _ in range(count):
            states.append(_make_state(doc_id, rng, templates))
            doc_id += 1
        path = data_dir / f"{name}.jsonl"
        save_dataset(states, path)
        print(f"  {name}: {len(states)} states → {path}")


def _make_state(doc_id: int, rng: Random, templates: tuple[str, ...]) -> TextState:
    """Sample latents, render one document, return TextState with gold labels."""
    latents = Latents(
        sentiment=rng.choice(SENTIMENTS),
        urgency=rng.choice(URGENCIES),
        quality=rng.randint(0, MAX_QUALITY),
        actionable=rng.random() < 0.5,
        pii=rng.random() < 0.5,
    )
    state_type = rng.choice(templates)
    render = _RENDERERS[state_type]
    text = render(rng, latents)
    return TextState(doc_id=doc_id, state_type=state_type, text=text, labels=latents.labels())


# --------------------------------------------------------------------------
# Shared rendering helpers
# --------------------------------------------------------------------------

def _person(rng: Random) -> tuple[str, str, str]:
    """Fake identity: (full name, email, phone)."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    company = rng.choice(COMPANIES).lower()
    return (
        f"{first} {last}",
        f"{first.lower()}.{last.lower()}@{company}.example.com",
        f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
    )


def _sentiment_phrase(rng: Random, sentiment: str) -> str:
    positive = (
        "Really pleased with how smoothly everything went — great work all around.",
        "Thanks for the excellent support, this resolved our issue perfectly.",
        "The team did a fantastic job; we are very happy with the results.",
    )
    negative = (
        "This is unacceptable and honestly quite frustrating to deal with.",
        "We are disappointed — the service has been broken for days.",
        "Very unhappy with the outcome; this fell far short of expectations.",
    )
    neutral = (
        "This is an informational update regarding the current status.",
        "Summary of the latest changes is provided below for reference.",
        "Noting the following facts for the record; no assessment implied.",
    )
    bank = {"positive": positive, "negative": negative, "neutral": neutral}[sentiment]
    return rng.choice(bank)


def _urgency_words(urgency: str) -> str:
    return {
        "low": "no rush",
        "medium": "when you get a chance",
        "high": "needs attention this week",
        "critical": "must be addressed immediately",
    }[urgency]


def _action_sentence(actionable: bool) -> str:
    if actionable:
        return "Action required: please review and approve the request below."
    return "This is informational only — no action is needed on your side."


def _contact_line(rng: Random, pii: bool) -> str:
    if pii:
        name, email, phone = _person(rng)
        return f"You can reach {name} at {email} or {phone}."
    return "Contact details have been redacted; reply via the support portal."


# --------------------------------------------------------------------------
# In-distribution templates
# --------------------------------------------------------------------------

def _render_email(rng: Random, lat: Latents) -> str:
    subject = {
        "low": f"Update: weekly digest ({_urgency_words(lat.urgency)})",
        "medium": f"Follow-up: pending request ({_urgency_words(lat.urgency)})",
        "high": f"Action required: pending approval ({_urgency_words(lat.urgency)})",
        "critical": f"URGENT: outage escalation ({_urgency_words(lat.urgency)})",
    }[lat.urgency]

    greeting = f"Hi team,\n\n" if lat.quality >= 1 else ""
    body = _sentiment_phrase(rng, lat.sentiment)
    if lat.quality == 0:
        body = body.split(".")[0][:40].lower()  # fragment with truncation artifacts
        parts = [body, _contact_line(rng, lat.pii)]
        return f"Subject: {subject}\n\n" + "\n".join(parts)

    parts = [f"Subject: {subject}\n\n", greeting, body]
    if lat.quality >= 2:
        parts.append(f"\nTimeline: {_urgency_words(lat.urgency)}.")
        parts.append(_action_sentence(lat.actionable))
    else:
        parts.append(_action_sentence(lat.actionable))
    if lat.quality == 3:
        parts.append("\nNext steps:\n- Owner assigned\n- Checkpoint scheduled\n- Metrics tracked")
    parts.append("\n" + _contact_line(rng, lat.pii))
    return "".join(parts)


def _render_ticket(rng: Random, lat: Latents) -> str:
    ticket_id = rng.randint(10000, 99999)
    title = {
        "low": f"Question about billing cycle ({_urgency_words(lat.urgency)})",
        "medium": f"Feature request: export options ({_urgency_words(lat.urgency)})",
        "high": f"Degraded performance reported ({_urgency_words(lat.urgency)})",
        "critical": f"SEV-1: service down for customers ({_urgency_words(lat.urgency)})",
    }[lat.urgency]

    desc = _sentiment_phrase(rng, lat.sentiment)
    if lat.quality >= 2:
        desc += f" SLA target: {_urgency_words(lat.urgency)}."
    if lat.quality == 0:
        desc = desc.split(".")[0].lower()

    if lat.pii:
        name, email, _ = _person(rng)
        requester = f"{name} <{email}>"
    else:
        requester = "[redacted]"

    resolution = (
        "Requested resolution: escalate and respond."
        if lat.actionable
        else "Resolution: none requested — documentation only."
    )
    return (
        f"Ticket #{ticket_id} [priority={lat.urgency}]\n"
        f"Title: {title}\n"
        f"Description: {desc}\n"
        f"{resolution}\n"
        f"Requester: {requester}"
    )


def _render_json(rng: Random, lat: Latents) -> str:
    import json as _json

    quality_words = ("incomplete", "minimal", "adequate", "thorough")
    doc = {
        "record_id": f"R-{rng.randint(1000, 9999)}",
        "summary": _sentiment_phrase(rng, lat.sentiment),
        "priority": lat.urgency,
        "timeline": _urgency_words(lat.urgency),
        "completeness": quality_words[lat.quality],
        "request_action": lat.actionable,
        "customer": (
            {k: v for k, v in zip(("name", "email", "phone"), _person(rng))}
            if lat.pii
            else "redacted"
        ),
    }
    return _json.dumps(doc, indent=2)


def _render_text(rng: Random, lat: Latents) -> str:
    note = f"Note {rng.randint(100, 999)}: {_sentiment_phrase(rng, lat.sentiment)}"
    lines = [note]
    if lat.quality >= 1:
        lines.append(f"Handling guidance: {_urgency_words(lat.urgency)}.")
        lines.append(_action_sentence(lat.actionable))
    if lat.quality == 3:
        lines.append("Checklist: triaged, documented, reviewed, archived.")
    lines.append(_contact_line(rng, lat.pii))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Held-out templates (unseen state types)
# --------------------------------------------------------------------------

def _render_report(rng: Random, lat: Latents) -> str:
    report_id = rng.randint(1, 99)
    findings = _sentiment_phrase(rng, lat.sentiment)
    if lat.pii:
        name, email, _ = _person(rng)
        prepared = f"Prepared by {name} ({email})"
    else:
        prepared = "Prepared by [analyst, identity withheld]"
    disposition = (
        "Recommendation: follow-up required."
        if lat.actionable
        else "Recommendation: file for reference, no follow-up."
    )
    return (
        f"WEEKLY REPORT #{report_id}\n"
        f"Priority assessment: {lat.urgency} ({_urgency_words(lat.urgency)})\n"
        f"Findings: {findings}\n"
        f"Data completeness: {_quality_words(lat.quality)}\n"
        f"{disposition}\n"
        f"{prepared}"
    )


def _render_log_entry(rng: Random, lat: Latents) -> str:
    severity = {"low": "INFO", "medium": "NOTICE", "high": "WARN", "critical": "ERROR"}[lat.urgency]
    stamps = ("2026-09-14 08:23:41", "2026-09-14 11:07:09", "2026-09-15 16:45:02")
    svc = rng.choice(("svc-api", "svc-auth", "svc-billing"))
    msg = _sentiment_phrase(rng, lat.sentiment)
    if lat.pii:
        user = f"user={rng.choice(FIRST_NAMES).lower()}.{rng.choice(LAST_NAMES).lower()}@mail.example.com"
    else:
        user = "user=[masked]"
    intervention = (
        "manual intervention required"
        if lat.actionable
        else "no operator action required"
    )
    detail = "" if lat.quality == 0 else f" detail_level={lat.quality}/3 (trace {_quality_words(lat.quality)})"
    return (
        f"{rng.choice(stamps)} {severity} [{svc}] {msg} "
        f"{user} latency={rng.randint(20, 900)}ms "
        f"{intervention} urgency={lat.urgency}{detail}"
    )


def _render_config_file(rng: Random, lat: Latents) -> str:
    name, email, phone = _person(rng)
    contact = (
        f"# maintainer: {name} <{email}> / {phone}"
        if lat.pii
        else "# maintainer: [redacted] — contact via on-call rotation"
    )
    feedback = _sentiment_phrase(rng, lat.sentiment)
    return (
        "# service configuration (annotated)\n"
        f"priority: {lat.urgency}          # {_urgency_words(lat.urgency)}\n"
        f"request_action: {str(lat.actionable).lower()}     # escalation policy\n"
        f"audit_completeness: \"{_quality_words(lat.quality)}\"  # record quality rubric\n"
        f"notes: \"{feedback}\"\n"
        f"{contact}"
    )


def _render_chat_message(rng: Random, lat: Latents) -> str:
    """Simulate a multi-turn chat conversation."""
    agent_name = rng.choice(("Alex", "Sam", "Taylor"))
    customer_name = rng.choice(FIRST_NAMES)
    ts = rng.choice(("10:14", "14:37", "09:52"))

    sentiment_line = _sentiment_phrase(rng, lat.sentiment)
    urgency_note = f"Priority: {lat.urgency.upper()}" if lat.urgency in ("high", "critical") else ""

    lines = [
        f"[{ts}] {agent_name} (support): Hi {customer_name}, how can I help?",
        f"[{ts}] {customer_name}: {sentiment_line}",
    ]
    if urgency_note:
        lines.append(f"[{ts}] {agent_name} (support): Noted — {urgency_note}. {_action_sentence(lat.actionable)}")
    else:
        lines.append(f"[{ts}] {agent_name} (support): Thanks for the update. {_action_sentence(lat.actionable)}")

    if lat.quality >= 2:
        lines.append(f"[{ts}] {agent_name} (support): I've logged this at detail level {lat.quality}/3.")
    if lat.pii:
        name, email, _ = _person(rng)
        lines.append(f"[{ts}] System: ticket created for {name} ({email})")
    else:
        lines.append(f"[{ts}] System: anonymous ticket created")
    return "\n".join(lines)


def _render_markdown(rng: Random, lat: Latents) -> str:
    """Render as a structured Markdown document."""
    qw = _quality_words(lat.quality)
    sentiment = _sentiment_phrase(rng, lat.sentiment)
    action_check = "[x]" if lat.actionable else "[ ]"

    lines = [
        f"# Status Update",
        f"",
        f"**Urgency:** `{lat.urgency.upper()}`",
        f"**Quality:** {qw} ({lat.quality}/3)",
        f"",
        f"## Summary",
        f"{sentiment}",
        f"",
        f"## Action Items",
        f"- {action_check} Follow-up required",
    ]
    if lat.quality >= 2:
        lines.append(f"- [x] Documentation updated")
        lines.append(f"- [x] Review checklist completed")
    if lat.pii:
        _, email, phone = _person(rng)
        lines.extend(["", "## Contact", f"- Email: `{email}`", f"- Phone: {phone}"])
    else:
        lines.extend(["", "## Contact", "- [redacted] — use support portal"])
    return "\n".join(lines)


def _render_bullet_list(rng: Random, lat: Latents) -> str:
    """Render as a terse bullet-point summary."""
    sentiment = _sentiment_phrase(rng, lat.sentiment)
    qw = _quality_words(lat.quality)
    urgency = _urgency_words(lat.urgency)

    lines = [
        f"• Status: {sentiment.split('.')[0]}",
        f"• Priority: {lat.urgency} ({urgency})",
        f"• Quality: {qw}",
        f"• Action needed: {'yes' if lat.actionable else 'no'}",
    ]
    if lat.quality >= 2:
        lines.append(f"• Detail level: {lat.quality}/3")
    if lat.pii:
        name, email, _ = _person(rng)
        lines.append(f"• Contact: {name} — {email}")
    else:
        lines.append("• Contact: [redacted]")
    return "\n".join(lines)


def _quality_words(quality: int) -> str:
    return ("incomplete", "minimal", "adequate", "thorough")[quality]


_RENDERERS = {
    "email": _render_email,
    "ticket": _render_ticket,
    "json": _render_json,
    "text": _render_text,
    "log_entry": _render_log_entry,
    "chat_message": _render_chat_message,
    "markdown": _render_markdown,
    "bullet_list": _render_bullet_list,
    "report": _render_report,
    "config_file": _render_config_file,
}