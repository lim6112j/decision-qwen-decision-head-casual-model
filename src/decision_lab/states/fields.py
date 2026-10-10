"""Field-level state splitting for set-based decision heads.

A state's text is decomposed into a set of field strings (sentences,
key:value lines, JSON leaves) so the head can attend over fields instead
of one pooled vector. Two sources of fields:

1. Renderer-emitted: ``TextState.fields`` set at generation time
   (Breakout renderers produce fields from the same latents as the text).
2. Generic heuristic: ``split_state_fields`` for arbitrary text — used for
   the fleet datasets (8 formats) so no regeneration is needed, and as the
   runtime fallback for caller-supplied states.

Fallback invariant: a state always yields at least one field (``[text]``),
so single-field sets reproduce the old pooled-vector behavior.
"""

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from decision_lab.states.dataset import TextState

MAX_FIELDS = 16

# Bumped whenever splitting behavior changes output for existing texts, so
# feature-cache fingerprints (backbone/features.py) invalidate: v2 added the
# degenerate-set guard (_split_single_part) in state_field_set.
SPLITTER_VERSION = 2

# Sentence boundary: ., !, ? followed by whitespace
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Clause boundary inside a sentence: comma, semicolon, colon
_CLAUSE_RE = re.compile(r"[,;:]\s+")
# KEY=VALUE log token (no internal whitespace)
_LOG_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*=[^\s]+")


def _split_single_part(text: str) -> list[str]:
    """Split a one-part text into ≥2 distinct fields, or return ``[text]``.

    Guard against the degenerate field set: when every field is identical
    (a single sentence, where the summary field duplicates the only field),
    the v2 head cannot discriminate options — identical attention keys give
    uniform attention, so every option reads the same context vector and
    the logits come out exactly equal. Clause boundaries (most natural
    break inside a sentence) are tried first, then a half-split into 2 word
    chunks. A text with a single word cannot be split at all — ``[text]``
    is returned and the set stays degenerate; nothing finer exists.
    """
    clauses = [part.strip() for part in _CLAUSE_RE.split(text) if part.strip()]
    if len(clauses) >= 2:
        return _merge_into_chunks(clauses, MAX_FIELDS)
    words = text.split()
    if len(words) < 2:
        return [text]
    size = -(-len(words) // 2)  # ceil division: exactly 2 chunks
    return [" ".join(words[:size]), " ".join(words[size:])]


def _flatten_json(value, prefix: str = "") -> list[str]:
    """Flatten nested JSON into "path.to.key: value" strings."""
    fields: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            fields.extend(_flatten_json(child, path))
    elif isinstance(value, list):
        rendered = ", ".join(str(item) for item in value)
        fields.append(f"{prefix}: {rendered}")
    else:
        fields.append(f"{prefix}: {value}")
    return fields


def _merge_into_chunks(fields: list[str], max_fields: int) -> list[str]:
    """Merge consecutive fields into at most max_fields even chunks.

    Preserves all content (nothing truncated) while capping set size so
    attention cost and the feature cache stay bounded.
    """
    n = len(fields)
    if n <= max_fields:
        return fields
    chunk_size = -(-n // max_fields)  # ceil division
    merged = [
        " ".join(fields[i : i + chunk_size])
        for i in range(0, n, chunk_size)
    ]
    return merged[:max_fields]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (after . ! ?), dropping empties."""
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


def split_state_fields(text: str) -> list[str]:
    """Split arbitrary state text into a field set.

    Strategy (first that yields >= 2 non-empty parts wins):
      1. JSON object → flatten to "path: value" leaves.
      2. Newlines → one field per non-empty line.
      3. KEY=VALUE log tokens → one field per token.
      4. Sentences (split after . ! ?).

    Single-part results fall back to ``[text]`` (whole text as one field).
    Output is capped at ``MAX_FIELDS`` by merging consecutive fields.

    Consistency invariant: training feature extraction and inference MUST
    resolve states through the same strategy — this function is the single
    source of truth (renderer-emitted ``fields`` bypass it only when the
    stored fields were produced by the same renderer at both times).
    """
    s = text.strip()
    if not s:
        return [""]

    # 1. JSON object
    if s.startswith("{"):
        try:
            fields = _flatten_json(json.loads(s))
            if len(fields) >= 2:
                return _merge_into_chunks(fields, MAX_FIELDS)
        except (json.JSONDecodeError, ValueError):
            pass  # malformed JSON → fall through to newline/sentence split

    # 2. Newline-delimited
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if len(lines) >= 2:
        return _merge_into_chunks(lines, MAX_FIELDS)

    # 3. KEY=VALUE log tokens (matches the breakout log template exactly)
    log_tokens = _LOG_TOKEN_RE.findall(s)
    if len(log_tokens) >= 2 and "".join(log_tokens).count("=") == len(log_tokens) and \
            s.replace(" ", "") == "".join(log_tokens):
        return _merge_into_chunks(log_tokens, MAX_FIELDS)

    # 4. Sentences
    sentences = split_sentences(s)
    if len(sentences) >= 2:
        return _merge_into_chunks(sentences, MAX_FIELDS)

    return [s]


def state_fields(state: "TextState") -> list[str]:
    """Field set for a state: explicit fields → heuristic split → [text]."""
    if state.fields:
        return list(state.fields)
    return split_state_fields(state.text)


def state_field_set(state: "TextState", include_summary: bool) -> list[str]:
    """Field set with an optional summary field (the full text) prepended.

    The summary guarantees M >= 1, gives the head global context, and
    mitigates distribution shift from heuristic field splitting.

    Degenerate-set guard: a set whose fields are all identical (single
    sentence + its own summary) gives uniform attention and identical
    logits for every option — appending a clause/word split restores ≥2
    distinct fields. Only single-token texts stay degenerate; there is
    nothing finer to split into.
    """
    fields = state_fields(state)
    if include_summary:
        fields = [state.text] + fields
    if len(set(fields)) < 2:
        existing = set(fields)
        fields = fields + [
            part for part in _split_single_part(state.text) if part not in existing
        ]
    return fields
