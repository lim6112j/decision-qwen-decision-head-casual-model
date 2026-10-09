#!/usr/bin/env python3
"""Verify the field-emitting breakout renderers are byte-identical to v1.

Regenerates the breakout datasets with the same seed and compares every
text against the existing JSONL on disk. Also prints field-set stats.
Exits 1 on any mismatch.
"""

import json
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from decision_lab.config import load_config
from decision_lab.states.breakout import make_breakout_state
from decision_lab.states.fields import split_state_fields, state_fields

DATA_DIR = Path("data")
SPLITS = {
    "train_breakout": 4000,
    "test_breakout": 500,
}


def main() -> int:
    cfg = load_config("configs/default.yaml")
    rng = Random(cfg.generator.seed + 1)

    ok = True
    doc_id = 0
    field_counts: dict[str, int] = {}
    for split, count in SPLITS.items():
        path = DATA_DIR / f"{split}.jsonl"
        existing = [json.loads(line) for line in path.read_text().strip().splitlines() if line]
        assert len(existing) == count, f"{path}: expected {count} rows, got {len(existing)}"
        for row in existing:
            state = make_breakout_state(doc_id, rng)
            doc_id += 1
            if state.text != row["text"]:
                ok = False
                print(f"MISMATCH {split} doc_id={row['doc_id']} ({state.state_type}):")
                print(f"  new: {state.text!r}")
                print(f"  old: {row['text']!r}")
            fields = state_fields(state)
            # Chunking-consistency invariant: heuristic split of the text
            # must reproduce the renderer's field set exactly, so a caller
            # that omits fields gets the same chunking as training.
            if state.fields is not None and split_state_fields(state.text) != fields:
                ok = False
                print(f"SPLIT MISMATCH {split} doc_id={row['doc_id']} ({state.state_type}):")
                print(f"  renderer: {fields!r}")
                print(f"  heuristic: {split_state_fields(state.text)!r}")
            field_counts[state.state_type] = field_counts.get(state.state_type, 0) + len(fields)
            if not fields or len(fields) > 16:
                ok = False
                print(f"BAD FIELD COUNT {split} doc_id={row['doc_id']}: {len(fields)}")

    if not ok:
        print("\nFAILED: renderer outputs drifted")
        return 1

    total = sum(field_counts.values())
    print("byte-identical: OK")
    print("fields per template (avg):")
    for t, c in sorted(field_counts.items()):
        n = SPLITS["train_breakout"] + SPLITS["test_breakout"]
        # rough: count/total_states_of_template — recompute properly below
        print(f"  {t}: total fields {c}")
    print(f"total fields: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
