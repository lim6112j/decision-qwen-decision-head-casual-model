"""Label store for real API traffic: pending queue + append-only labels.

A logged ``/api/decide-dynamic`` call produces one :class:`LabelItem` per
(call, question) pair. Items are deduped by a content hash so the same
(state, question, options) sent twice is labeled once. Labels are appended
to a JSONL file, never rewritten, and appends are idempotent by ``item_id``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# How a label was produced. ``accepted``/``overridden`` are manual (the UI
# pre-fills the model's prediction); ``auto`` is the OpenRouter labeler.
SOURCE_ACCEPTED = "accepted"
SOURCE_OVERRIDDEN = "overridden"
SOURCE_AUTO = "auto"
VALID_SOURCES = (SOURCE_ACCEPTED, SOURCE_OVERRIDDEN, SOURCE_AUTO)

_APPEND_LOCK = threading.Lock()
_ID_SEP = "\x1e"    # field separator inside the hash
_OPT_SEP = "\x1f"   # separator inside the option list


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def option_texts(question: dict) -> list[str]:
    """The ordered label space for a question config.

    choice → ``options``; score → ``levels`` (index = rank); noul → the
    fixed ``["false", "true"]`` order the head is trained/served with.
    """
    kind = question.get("type")
    if kind == "noul":
        return ["false", "true"]
    if kind == "choice":
        return list(question.get("options") or [])
    if kind == "score":
        return list(question.get("levels") or [])
    raise ValueError(f"unknown question type {kind!r}")


def predicted_index(question: dict, answer: dict | None) -> int | None:
    """Map a decoded API answer back to an index into ``option_texts``.

    ``predicted`` is a bool (noul), an option string (choice), or a 0-based
    level index (score) — see docs/http-api.md. Returns None when the answer
    is missing or can't be mapped.
    """
    if answer is None:
        return None
    pred = answer.get("predicted")
    kind = question.get("type")
    try:
        if kind == "noul":
            return 1 if pred else 0
        if kind == "choice":
            return option_texts(question).index(pred)
        return int(pred)
    except (ValueError, TypeError):
        return None


def make_item_id(
    text: str, fields: list[str] | None, question: str, options: list[str], kind: str,
) -> str:
    """Stable content hash identifying a labelable (state, question) pair."""
    h = hashlib.sha256()
    for part in (text, _OPT_SEP.join(fields or []), question or "", _OPT_SEP.join(options), kind):
        h.update(part.encode("utf-8"))
        h.update(_ID_SEP.encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class LabelItem:
    """One labelable (call, question) pair, with its label once produced."""

    item_id: str
    call_id: str
    text: str
    question: str          # question text (may be "")
    kind: str              # choice | score | noul
    options: list[str]     # ordered label space
    predicted_idx: int | None = None   # the head's answer, for pre-fill
    fields: list[str] | None = None    # caller chunking (custom_fields)
    doc_id: int | None = None
    state_type: str = "custom"
    # label fields
    gold_idx: int | None = None
    source: str | None = None
    labeled_by: str = ""
    labeled_at: str = ""

    @property
    def labeled(self) -> bool:
        return self.gold_idx is not None and self.source is not None

    def gold_label(self) -> str | None:
        if self.gold_idx is None or not (0 <= self.gold_idx < len(self.options)):
            return None
        return self.options[self.gold_idx]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LabelItem":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def items_from_call(row: dict) -> list[LabelItem]:
    """Explode one traffic-log row into labelable items (skips malformed)."""
    text = row.get("text") or ""
    fields = row.get("fields")
    questions = row.get("questions") or []
    answers = row.get("answers") or []
    items: list[LabelItem] = []

    for qi, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        try:
            opts = option_texts(q)
        except ValueError:
            continue
        if len(opts) < 2:
            continue
        qt = (q.get("question") or "").strip()
        kind = q["type"]
        items.append(LabelItem(
            item_id=make_item_id(text, fields, qt, opts, kind),
            call_id=row.get("call_id", ""),
            text=text,
            question=qt,
            kind=kind,
            options=opts,
            fields=fields,
            predicted_idx=predicted_index(q, answers[qi] if qi < len(answers) else None),
            doc_id=row.get("doc_id"),
            state_type=row.get("state_type", "custom"),
        ))
    return items


def read_traffic(traffic_dir: Path) -> list[dict]:
    """Read every row across the traffic JSONL files (sorted by filename)."""
    rows: list[dict] = []
    if not traffic_dir.exists():
        return rows
    for path in sorted(traffic_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue   # a corrupt line must not sink the whole queue
    return rows


def load_labels(labels_path: Path) -> dict[str, LabelItem]:
    """Load the label file → {item_id: LabelItem} (last row wins)."""
    labeled: dict[str, LabelItem] = {}
    if not labels_path.exists():
        return labeled
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = LabelItem.from_dict(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
        labeled[item.item_id] = item
    return labeled


def load_discarded(discarded_path: Path | None) -> set[str]:
    """Item ids removed from the queue without being labeled."""
    if discarded_path is None or not discarded_path.exists():
        return set()
    ids: set[str] = set()
    for line in discarded_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["item_id"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return ids


def discard_item(item_id: str, discarded_path: Path) -> bool:
    """Remove an item from the queue without labeling it. Idempotent."""
    with _APPEND_LOCK:
        if item_id in load_discarded(discarded_path):
            return False
        discarded_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"item_id": item_id, "discarded_at": _now_iso()}
        with discarded_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True


def build_queue(
    traffic_dir: Path, labels_path: Path, discarded_path: Path | None = None,
) -> list[LabelItem]:
    """Pending items: unique traffic items not labeled and not discarded."""
    labeled = load_labels(labels_path)
    discarded = load_discarded(discarded_path)
    seen: dict[str, LabelItem] = {}
    for row in read_traffic(traffic_dir):
        for item in items_from_call(row):
            if item.item_id in seen or item.item_id in labeled or item.item_id in discarded:
                continue
            seen[item.item_id] = item
    return list(seen.values())


def append_label(item: LabelItem, labels_path: Path) -> bool:
    """Append a labeled item; idempotent by ``item_id``.

    Returns True if written, False if that ``item_id`` was already labeled.
    """
    with _APPEND_LOCK:
        if item.item_id in load_labels(labels_path):
            return False
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        if not item.labeled_at:
            item.labeled_at = _now_iso()
        with labels_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        return True


def stats(
    traffic_dir: Path, labels_path: Path, discarded_path: Path | None = None,
) -> dict:
    """Counts for the UI progress bar: total / labeled / discarded / remaining."""
    labeled = len(load_labels(labels_path))
    discarded = len(load_discarded(discarded_path))
    remaining = len(build_queue(traffic_dir, labels_path, discarded_path))
    total = labeled + discarded + remaining
    return {"total": total, "labeled": labeled, "discarded": discarded, "remaining": remaining}