"""Label-preserving text-shape variants for training-data augmentation.

Real callers send states in shapes the generator templates never render:
a browser UI that owns `{from, subject, body}` keys joins the values into
one bare line; scripts strip structural markers before sending. The v2/v3
field-set head is accurate on trained shapes but collapses to near-uniform
logits on untrained shapes (verified against the live head: the same
latents, flattened and marker-stripped, dropped urgency confidence 0.90 →
0.54 and flipped is_actionable).

Augmentation adds variants of each training text so the head learns shape
invariance instead of keying on template surface structure. Every variant
is a pure re-shaping: content that carries a latent signal is never
dropped, so gold labels stay exact by construction.

The transforms deliberately do NOT touch breakout states (game states are
already trained bare — and their gold convention is the opposite of the
documents': "situation:" prefixes measurably flip breakout answers while
restoring document confidence).
"""

import re

# "Label: content" — a line-lead structural marker ("Subject: ...",
# "**Urgency:** ...", "- Priority: ..."). The content after the colon is
# label-bearing; the marker word itself is not (gold comes from latents).
_LEAD_LABEL_RE = re.compile(r"^(?:[#•\-\s]*\*{0,2})([A-Za-z][A-Za-z _/-]{0,30})\*{0,2}\s*:\s*")
# Greeting-only line ("Hi team," / "Hello,") — no latent lives here.
_GREETING_RE = re.compile(r"^(?:Hi|Hello|Dear)[^.!?]{0,30},\s*$")
# Markdown residue: bold/inline-code fences and heading hashes.
_MD_RESIDUE_RE = re.compile(r"\*{1,2}|`+|^#+\s*", re.MULTILINE)


def _flatten(text: str) -> str:
    """Collapse all whitespace (newlines, indents) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _unmarked(text: str) -> str:
    """Flatten and strip structural markers, keeping their content.

    - "Subject: X" / "**Urgency:** X" / "• Status: X" → "X"
    - greeting-only lines ("Hi team,") → dropped
    - markdown bold/code/heading residue removed

    The result is the "bare values" shape real callers produce: no keys,
    no headers, no bullets — the shape that collapses the head today.
    """
    kept: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _GREETING_RE.match(line):
            continue
        line = _MD_RESIDUE_RE.sub("", line)
        m = _LEAD_LABEL_RE.match(line)
        if m:
            line = line[m.end():]
        line = line.strip()
        if line:
            kept.append(line)
    return " ".join(kept)


def shape_variants(text: str) -> list[str]:
    """Label-preserving shape variants of one text, deduped, original first.

    Returns up to 2 variants: flattened whitespace and marker-stripped
    flatten. Variants identical to the original (or each other) are
    dropped — a duplicate field set would be exactly the degenerate input
    state_field_set guards against.
    """
    variants: list[str] = []
    for candidate in (_flatten(text), _unmarked(text)):
        if candidate != text and candidate not in variants and candidate:
            variants.append(candidate)
    return variants
