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
from decision_lab.states.fields import state_field_set

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

USER_STATE_TEXT = (
    "The ball is clearly to the LEFT of the paddle (gap 124 px) and moving "
    "right and up, away from the paddle."
)
USER_QUESTION = "which direction the paddle move?"


def _embed_state(server, head, text: str, device, include_summary: bool) -> torch.Tensor:
    """Embed a state as a field set → (1, M, D) (v2) or (1, D) (v1).

    Chunking goes through split_state_fields/state_field_set — the same
    path as feature extraction — so inference chunks exactly like training.
    """
    from decision_lab.states.fields import split_state_fields

    if head.state_set:
        from decision_lab.states.dataset import TextState

        pseudo_state = TextState(doc_id=-1, state_type="custom", text=text, labels={})
        field_texts = state_field_set(pseudo_state, include_summary)
    else:
        field_texts = [text]
    embs = server.embed(field_texts)
    return torch.tensor(embs, dtype=torch.float32, device=device).unsqueeze(0)


def _predict(server, head, temperature, text: str, options: list[str], device,
             include_summary: bool, question_text: str = USER_QUESTION) -> dict:
    state_emb = _embed_state(server, head, text, device, include_summary)
    cache = {}
    embs = server.embed([*options, question_text])
    for t, emb in zip([*options, question_text], embs):
        cache[t] = torch.tensor(emb, dtype=torch.float32, device=device)
    questions = [{"type": "choice", "options": list(options), "question": question_text}]
    results = predict_dynamic(state_emb, questions, cache, head, temperature)
    return results[0]


def main():
    num_states = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    cfg = load_config(ROOT / "configs" / "default.yaml")
    device = get_device()

    head = load_dynamic_head(str(MODELS_DIR / "head_dynamic.pt"), device=device).eval()
    temperature = getattr(head, "temperature", 1.0)
    include_summary = cfg.dynamic_head.include_summary_field

    gguf = Path(cfg.model.gguf_path).expanduser().resolve()
    with LlamaServer(gguf, port=cfg.model.server_port, context_length=cfg.model.context_length) as server:
        # 1. The user's exact failing state
        print("=" * 70)
        print("Exact user state (expect 'left'):")
        print(f"  {USER_STATE_TEXT}")
        answer = _predict(server, head, temperature, USER_STATE_TEXT,
                          ["left", "right", "stay"], device, include_summary)
        dist = ", ".join(f"{k}={v:.3f}" for k, v in answer["distribution"].items())
        print(f"  → predicted: {answer['predicted']}  (confidence {answer['confidence']:.3f})")
        print(f"    distribution: {dist}")
        ok_base = answer["predicted"] == "left"

        # 2. Same question, shuffled option order
        answer_shuffled = _predict(server, head, temperature, USER_STATE_TEXT,
                                   ["stay", "right", "left"], device, include_summary)
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
                              ["left", "right", "stay"], device, include_summary)
            predicted_labels.add(answer["predicted"])
            mark = "✓" if answer["predicted"] == gold else "✗"
            print(f"  [{mark}] gold={gold:5s} pred={answer['predicted']:5s}"
                  f"  ({answer['confidence']:.2f})  {row['text'][:70]}")
        print(f"  distinct predictions: {sorted(predicted_labels)}")

        # 4. Question-flip probe (v3): one state, same options, two questions
        print("=" * 70)
        print("Question-flip probe (same state, same options, two questions):")
        row = rows[0]
        paddle_gold = row["labels"]["paddle_direction"]
        motion_gold = row["labels"]["ball_motion"]
        BALL_MOTION_QUESTION = "Which way is the ball moving horizontally?"
        answer_paddle = _predict(server, head, temperature, row["text"],
                                 ["left", "right", "stay"], device, include_summary)
        answer_motion = _predict(server, head, temperature, row["text"],
                                 ["left", "right", "stay"], device, include_summary,
                                 question_text=BALL_MOTION_QUESTION)
        print(f"  paddle_direction (gold={paddle_gold}): → {answer_paddle['predicted']}")
        print(f"  ball_motion      (gold={motion_gold}): → {answer_motion['predicted']}")
        ok_flip = (answer_paddle["predicted"] == paddle_gold
                   and answer_motion["predicted"] == motion_gold)
        if ok_flip:
            print("  → both questions match their own gold (question text is functional)")
        else:
            print("  → MISMATCH: the head is not conditioning on the question text")

    if not (ok_base and ok_shuffled):
        print("FAIL: exact user state did not predict 'left' in both option orders")
        sys.exit(1)
    if not ok_flip:
        print("FAIL: question-flip probe did not match both golds")
        sys.exit(1)
    print("PASS: exact user state predicts 'left' in both option orders + question flip works")


if __name__ == "__main__":
    main()
