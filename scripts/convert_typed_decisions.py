#!/usr/bin/env python
"""Convert `LocalLLaMA/typed-decisions` parquet → our question-format JSONL.

One output row per (case, question):

    {"id","workflow","qid","question_id","text","question":{...},"gold_idx","option_keys"}

`text` is the case's JSON state (our field splitter chunks JSON per leaf);
`question` is our serving-format config so the row can be fed straight to the
head. Used to measure the head on real typed-decision data (OOD).

**`qid` is NOT globally unique — it is only unique within a workflow.** Three
names collide across workflows and mean genuinely different questions:

    action       agent_trace_observability (4 opts) vs customer_service (5 opts)
    disposition  invoice_processing vs security_incidents
    urgency      all four workflows (four different phrasings)

15 distinct qids are therefore 19 distinct questions on the test split. Every
row additionally carries `question_id = "{workflow}:{qid}"`, which IS globally
unique, and that is what anything grouping "per question" must key on.

Their schema (verified 2026-10-13):
  choice.qdef.criteria = {option_key: description}          → options = descriptions
  score.qdef.criteria  = [level_description, ...]           → levels  = descriptions
  noul.qdef.criteria   = {"false":..,"true":..} or null     → boolean
  gold[qid].label      = option_key | "true"/"false" | level index (str)

Requires pyarrow (not a project dep):

    uv run --with pyarrow python scripts/convert_typed_decisions.py \\
        /tmp/typed-decisions/all_test.parquet -o /tmp/td_test.jsonl

The parquet is downloaded data, so it is passed as a path argument and read
from an isolated directory; this script lives in the repo, not next to it.
"""

from __future__ import annotations

import argparse
import json

import pyarrow.parquet as pq


def _question_config(qdef: dict) -> tuple[dict, list[str] | None] | None:
    """Their qdef → (our serving-format config, option keys or None)."""
    kind = qdef.get("type")
    instr = (qdef.get("instructions") or "").strip()
    crit = qdef.get("criteria")
    if kind == "choice":
        if not isinstance(crit, dict) or len(crit) < 2:
            return None
        keys = [str(k) for k in crit.keys()]
        return {"type": "choice", "options": [str(crit[k]) for k in keys],
                "question": instr}, keys
    if kind == "noul":
        return {"type": "noul", "question": instr}, None
    if kind == "score":
        if isinstance(crit, list):
            levels = [str(x) for x in crit]
        elif isinstance(crit, dict):
            levels = [str(v) for v in crit.values()]
        else:
            return None
        if len(levels) < 2:
            return None
        return {"type": "score", "levels": levels, "question": instr}, None
    return None


def _gold_index(kind: str, label, keys: list[str] | None) -> int | None:
    if kind == "choice":
        return keys.index(str(label)) if keys and str(label) in keys else None
    if kind == "noul":
        return 1 if str(label).lower() == "true" else 0
    try:
        return int(label)          # score: level index
    except (TypeError, ValueError):
        return None


def question_id(workflow: str, qid: str) -> str:
    """Globally-unique question name: workflow-scoped qid.

    `qid` alone is only unique within a workflow (see the module docstring),
    so `(workflow, qid)` is the identity of a question. The namespace is kept
    in the name so a bare qid is still recoverable via `rsplit(":", 1)[-1]`.
    """
    return f"{workflow}:{qid}"


def _labels_row(rec: dict) -> dict:
    """Our LabelItem-shaped row (for the real-traffic `build-real` pipeline)."""
    from decision_lab.real.labels import make_item_id

    q = rec["question"]
    kind = q["type"]
    options = q["options"] if kind == "choice" else (
        q["levels"] if kind == "score" else ["false", "true"])
    question = q.get("question") or rec["question_id"]
    return {
        "item_id": make_item_id(rec["text"], None, question, options, kind),
        "call_id": rec["id"], "text": rec["text"], "question": question,
        "kind": kind, "options": list(options), "gold_idx": rec["gold_idx"],
        "source": "accepted", "labeled_by": "typed-decisions",
    }


def convert(parquet_paths: list[str], out_path: str, labels_out: str | None = None) -> int:
    out: list[dict] = []
    for path in parquet_paths:
        for row in pq.read_table(path).to_pylist():
            questions = json.loads(row["questions"])
            gold = json.loads(row["gold"])
            for qid, qdef in questions.items():
                cfg_keys = _question_config(qdef)
                if cfg_keys is None:
                    continue
                cfg, keys = cfg_keys
                label = (gold.get(qid) or {}).get("label")
                gold_idx = _gold_index(cfg["type"], label, keys)
                if gold_idx is None:
                    continue
                rec = {
                    "id": row["id"], "workflow": row["workflow"], "qid": qid,
                    "question_id": question_id(row["workflow"], qid),
                    "text": row["state"], "question": cfg, "gold_idx": gold_idx,
                }
                if keys is not None:
                    rec["option_keys"] = keys
                out.append(rec)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if labels_out:
        with open(labels_out, "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(_labels_row(r), ensure_ascii=False) + "\n")
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquets", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--labels-out", default=None,
                    help="also write LabelItem rows (real-traffic labels.jsonl)")
    args = ap.parse_args()
    n = convert(args.parquets, args.out, args.labels_out)
    print(f"wrote {n} items → {args.out}")
    if args.labels_out:
        print(f"wrote {n} LabelItem rows → {args.labels_out}")


if __name__ == "__main__":
    main()
