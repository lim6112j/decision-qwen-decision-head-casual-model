"""Quick Breakout test: the exact state that used to always answer "right".

Standalone — starts llama-server, loads models/head_dynamic.pt, and probes:
  1. The user's exact paddle-control state (expect "left").
  2. The same question with shuffled option order (permutation robustness).
  3. ~20 random states from data/test_breakout.jsonl (all three labels appear).

Usage:
    .venv/bin/python scripts/quick_breakout_test.py [num_states]

Exits 1 unless the exact state predicts "left" in both option orders.
"""

import json
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from decision_lab.backbone.llama_server import LlamaServer
from decision_lab.config import load_config
from decision_lab.head.dynamic_model import (
    get_device,
    load_dynamic_head,
    predict_dynamic,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

USER_STATE_TEXT = (
    "The ball is clearly to the LEFT of the paddle (gap 124 px) and moving "
    "right and up, away from the paddle."
)
USER_QUESTION = "which direction the paddle move?"


def _embed_state(server, text: str, device) -> torch.Tensor:
    [emb] = server.embed([text])
    return torch.tensor(emb, dtype=torch.float32, device=device)


def _predict(server, head, temperature, text: str, options: list[str], device) -> dict:
    state_emb = _embed_state(server, text, device)
    cache = {}
    embs = server.embed(options)
    for opt, emb in zip(options, embs):
        cache[opt] = torch.tensor(emb, dtype=torch.float32, device=device)
    questions = [{"type": "choice", "options": list(options), "question": USER_QUESTION}]
    results = predict_dynamic(state_emb, questions, cache, head, temperature)
    return results[0]


def main():
    num_states = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = get_device()

    head = load_dynamic_head(str(MODELS_DIR / "head_dynamic.pt"), device=device).eval()
    temperature = getattr(head, "temperature", 1.0)

    gguf = Path(cfg.model.gguf_path).expanduser().resolve()
    with LlamaServer(gguf, port=cfg.model.server_port, context_length=cfg.model.context_length) as server:
        # 1. The user's exact failing state
        print("=" * 70)
        print("Exact user state (expect 'left'):")
        print(f"  {USER_STATE_TEXT}")
        answer = _predict(server, head, temperature, USER_STATE_TEXT,
                          ["left", "right", "stay"], device)
        dist = ", ".join(f"{k}={v:.3f}" for k, v in answer["distribution"].items())
        print(f"  → predicted: {answer['predicted']}  (confidence {answer['confidence']:.3f})")
        print(f"    distribution: {dist}")
        ok_base = answer["predicted"] == "left"

        # 2. Same question, shuffled option order
        answer_shuffled = _predict(server, head, temperature, USER_STATE_TEXT,
                                   ["stay", "right", "left"], device)
        print(f"  → shuffled option order: {answer_shuffled['predicted']}")
        ok_shuffled = answer_shuffled["predicted"] == "left"

        # 3. Random test states: predicted vs gold
        states_path = DATA_DIR / "test_breakout.jsonl"
        rows = [json.loads(line) for line in states_path.read_text().strip().splitlines() if line]
        rng = Random(0)
        rng.shuffle(rows)
        rows = rows[:num_states]

        print("=" * 70)
        print(f"{len(rows)} random states from test_breakout.jsonl:")
        predicted_labels = set()
        for row in rows:
            gold = row["labels"]["paddle_direction"]
            answer = _predict(server, head, temperature, row["text"],
                              ["left", "right", "stay"], device)
            predicted_labels.add(answer["predicted"])
            mark = "✓" if answer["predicted"] == gold else "✗"
            print(f"  [{mark}] gold={gold:5s} pred={answer['predicted']:5s}"
                  f"  ({answer['confidence']:.2f})  {row['text'][:70]}")
        print(f"  distinct predictions: {sorted(predicted_labels)}")

    if not (ok_base and ok_shuffled):
        print("FAIL: exact user state did not predict 'left' in both option orders")
        sys.exit(1)
    print("PASS: exact user state predicts 'left' in both option orders")


if __name__ == "__main__":
    main()
