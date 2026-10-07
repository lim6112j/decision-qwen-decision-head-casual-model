"""Text states with gold labels: dataclass + JSONL I/O."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextState:
    """A single text state with gold labels for every question in the bank.

    labels values by question type:
      choice → option key (str); score → level index (int); noul → bool.
    """

    doc_id: int
    state_type: str
    text: str
    labels: dict

    def render(self) -> str:
        return self.text


def save_dataset(states: list[TextState], path: Path) -> None:
    """Save states to JSONL (one row per state)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "doc_id": s.doc_id,
            "state_type": s.state_type,
            "text": s.text,
            "labels": s.labels,
        }
        for s in states
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))


def load_dataset(path: Path) -> list[TextState]:
    """Load JSONL into a list of TextState objects."""
    states = []
    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        d = json.loads(line)
        states.append(TextState(
            doc_id=d["doc_id"],
            state_type=d["state_type"],
            text=d["text"],
            labels=d["labels"],
        ))
    return states