#!/usr/bin/env python
"""Evaluate the dynamic head on converted `LocalLLaMA/typed-decisions` items.

OOD diagnostic: the head is trained on this repo's question bank, so these
real typed-decision questions are out-of-distribution — this measures how far
it generalizes to real decision data (as opposed to synthetic templates).

Needs a llama-server already running on `--port` (the same bge GGUF the head
was trained against). Reads the JSONL from convert_typed_decisions.py.

    llama-server -m models/bge-m3-Q8_0.gguf --port 8092 --embeddings \
        --pooling cls --no-webui &
    uv run python scripts/eval_typed_decisions.py /tmp/td_test.jsonl \
        --head models/head_dynamic_bge.pt --port 8092
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import requests
import torch

from decision_lab.head.dynamic_model import (
    is_correct_dynamic,
    load_dynamic_head,
    predict_dynamic,
    question_option_texts,
)
from decision_lab.states.dataset import TextState
from decision_lab.states.fields import state_field_set


def _embed(port: int, texts: list[str], batch: int = 32) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    uniq = list(dict.fromkeys(texts))
    for i in range(0, len(uniq), batch):
        chunk = uniq[i : i + batch]
        r = requests.post(
            f"http://127.0.0.1:{port}/v1/embeddings",
            json={"input": chunk}, timeout=120,
        )
        r.raise_for_status()
        for t, e in zip(chunk, r.json()["data"]):
            out[t] = np.array(e["embedding"], dtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--head", default="models/head_dynamic_bge.pt")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--include-summary", default="true")
    args = ap.parse_args()

    include_summary = args.include_summary.lower() == "true"
    head = load_dynamic_head(args.head)
    head.eval()

    items = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    # Build field sets (chunking must match training) + collect texts to embed.
    prepared = []
    texts: list[str] = []
    for it in items:
        st = TextState(doc_id=-1, state_type="custom", text=it["text"], labels={})
        fields = state_field_set(st, include_summary)
        prepared.append((it, fields))
        texts.extend(fields)
        texts.extend(question_option_texts(it["question"]))
        if it["question"].get("question"):
            texts.append(it["question"]["question"])

    print(f"embedding {len(set(texts))} unique texts via :{args.port} ...")
    emb = _embed(args.port, texts)
    key = lambda t: torch.from_numpy(emb[t])

    correct = defaultdict(int)
    total = defaultdict(int)
    n_ok = 0
    for it, fields in prepared:
        state = torch.stack([key(t) for t in fields])
        cache = {t: key(t) for t in question_option_texts(it["question"])}
        if it["question"].get("question"):
            cache[it["question"]["question"]] = key(it["question"]["question"])
        pred = predict_dynamic(state, [it["question"]], cache, head)[0]
        # is_correct_dynamic compares choice on TEXT, score/noul on index/bool.
        gold = it["gold_idx"]
        if it["question"]["type"] == "choice":
            gold = it["question"]["options"][int(gold)]
        ok = is_correct_dynamic(it["question"], pred["predicted"], gold)
        n_ok += ok
        # Per-question buckets key on question_id: the raw qid is only unique
        # WITHIN a workflow (e.g. `urgency` exists in all four workflows as
        # four different questions), so bucketing by qid would silently merge
        # them. The name is namespaced ("{workflow}:{qid}") but still reads
        # as the qid it came from (see convert_typed_decisions.py).
        question_id = it.get("question_id") or f"{it['workflow']}:{it['qid']}"
        for k in ("__all__", it["question"]["type"], question_id, it["workflow"]):
            total[k] += 1
            correct[k] += ok

    print(f"\nOVERALL: {n_ok}/{len(items)} = {100 * n_ok / len(items):.1f}%  (OOD)\n")
    for label_, keys in (
        ("type", sorted({it["question"]["type"] for it in items})),
        ("workflow", sorted({it["workflow"] for it in items})),
        ("question_id", sorted({it.get("question_id") or
                                f"{it['workflow']}:{it['qid']}" for it in items})),
    ):
        print(f"-- by {label_} --")
        for k in keys:
            print(f"   {k:28} {correct[k]:4}/{total[k]:4} = {100 * correct[k] / total[k]:5.1f}%")
        print()


if __name__ == "__main__":
    main()
