"""Quick one-state test: ask each model the same typed questions over one state.

Standalone — does not modify any existing code. Usage:
    .venv/bin/python scripts/quick_one_state_test.py [num_states]

Prints each model's predicted answers vs ground truth, plus latency per decision.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from decision_lab.backbone.llama_server import LlamaServer
from decision_lab.config import load_config
from decision_lab.head.model import (
    build_question_spec,
    create_random_head,
    get_device,
    load_head,
    predict_all,
)
from decision_lab.prompt_lm.agent import PromptAgent, PromptMode

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def _format_answers(decoded: dict) -> str:
    parts = []
    for qid, d in decoded.items():
        pred = d["predicted"]
        shown = str(pred).lower() if isinstance(pred, bool) else pred
        parts.append(f"{qid}={shown} ({d['confidence']:.2f})")
    return "  ".join(parts)


def main():
    num_states = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cfg = load_config(ROOT / "configs" / "default.yaml")
    question_spec = build_question_spec(cfg.questions)

    # One state per test set, read straight from JSONL (no caches needed)
    rows_by_split = {}
    for split in ["test_indist", "test_heldout"]:
        with open(DATA_DIR / f"{split}.jsonl") as f:
            rows_by_split[split] = [json.loads(next(f)) for _ in range(num_states)]

    gguf = Path(cfg.model.gguf_path).expanduser().resolve()
    device = get_device()

    head_trained = load_head(str(MODELS_DIR / "head_trained.pt"), device=device).eval()
    head_random = create_random_head(question_spec).to(device).eval()

    with LlamaServer(gguf, port=cfg.model.server_port, context_length=cfg.model.context_length) as server:
        agent_zero = PromptAgent(cfg, server, mode=PromptMode.ZERO_SHOT)

        for split, rows in rows_by_split.items():
            print(f"\n{'=' * 60}\n{split} — {len(rows)} state(s)\n{'=' * 60}")
            for row in rows:
                gold = row["labels"]
                print(f"\nState: {row['state_type']} (doc {row['doc_id']})")
                print(row["text"])
                print(f"  gold: {gold}")
                print("-" * 40)

                # head_trained / head_random: embed the state text, single forward pass
                embed_t0 = time.perf_counter()
                feat = np.array(server.embed([row["text"]]), dtype=np.float32)
                embed_ms = (time.perf_counter() - embed_t0) * 1000
                x = torch.tensor(feat, device=device)

                for agent_id, head in [("head_trained", head_trained), ("head_random", head_random)]:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        decoded = predict_all(head.question_spec, head(x), head.temperatures)
                    ms = (time.perf_counter() - t0) * 1000
                    print(f"  [{agent_id:14s}] {_format_answers(decoded)}  "
                          f"| head {ms:.1f} ms (+ embed {embed_ms:.0f} ms)")

                # prompt_lm_zero_shot — chat completion answers in text
                class ShimState:
                    def render(self):
                        return row["text"]

                t0 = time.perf_counter()
                answers, raw = agent_zero.decide(ShimState())
                ms = (time.perf_counter() - t0) * 1000
                shown = "  ".join(f"{qid}={ans}" for qid, ans in answers.items())
                print(f"  [{'prompt_lm_zero_shot':14s}] {shown or 'PARSE-FAIL'}  | {ms:.0f} ms")
                print(f"    raw: {raw!r}")


if __name__ == "__main__":
    main()