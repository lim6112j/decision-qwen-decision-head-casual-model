"""Quick one-state test: ask each model the same single gridworld question.

Standalone — does not modify any existing code. Usage:
    .venv/bin/python scripts/quick_one_state_test.py [num_states]

Prints each model's predicted action vs ground truth, plus latency per decision.
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
from decision_lab.env.gridworld import ACTION_NAMES
from decision_lab.head.model import DecisionHead, get_device, load_head
from decision_lab.prompt_lm.agent import PromptAgent, PromptMode

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def main():
    num_states = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cfg = load_config(ROOT / "configs" / "default.yaml")

    # One question per test set, read straight from JSONL (no caches needed)
    questions = {}
    for split in ["test_indist", "test_heldout"]:
        with open(DATA_DIR / f"{split}.jsonl") as f:
            rows = [json.loads(next(f)) for _ in range(num_states)]
        questions[split] = rows

    gguf = Path(cfg.model.gguf_path).expanduser().resolve()
    device = get_device()

    head_trained = load_head(
        str(MODELS_DIR / "head_trained.pt"),
        hidden_dim=cfg.head.hidden_dim,
        num_actions=cfg.head.num_actions,
        dropout=cfg.head.dropout,
    ).to(device).eval()

    head_random = DecisionHead(
        hidden_dim=cfg.head.hidden_dim,
        num_actions=cfg.head.num_actions,
        dropout=cfg.head.dropout,
    ).to(device).eval()

    with LlamaServer(gguf, port=cfg.model.server_port, context_length=cfg.model.context_length) as server:
        agent_zero = PromptAgent(cfg, server, mode=PromptMode.ZERO_SHOT)

        for split, rows in questions.items():
            print(f"\n{'=' * 60}\n{split} — {len(rows)} question(s)\n{'=' * 60}")
            for row in rows:
                gt = row["action"]
                print(f"\nQuestion: agent={row['agent_pos']} goal={row['goal_pos']} "
                      f"| ground truth: {ACTION_NAMES[gt]} ({gt})")
                print(row["text"])
                print("-" * 40)

                # head_trained / head_random: embed the state text, then predict
                embed_t0 = time.perf_counter()
                feat = np.array(server.embed([row["text"]]), dtype=np.float32)
                embed_ms = (time.perf_counter() - embed_t0) * 1000
                x = torch.tensor(feat, device=device)

                for agent_id, head in [("head_trained", head_trained), ("head_random", head_random)]:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        pred = head(x).argmax(dim=1).item()
                    ms = (time.perf_counter() - t0) * 1000
                    verdict = "OK" if pred == gt else "MISS"
                    print(f"  [{agent_id:14s}] answer: {ACTION_NAMES[pred]} ({pred})  "
                          f"{verdict}  | head {ms:.1f} ms (+ embed {embed_ms:.0f} ms)")

                # prompt_lm_zero_shot — reuse the JSONL text directly via a tiny shim
                class ShimState:
                    def render(self):
                        return row["text"]

                t0 = time.perf_counter()
                action, raw = agent_zero.decide(ShimState())
                ms = (time.perf_counter() - t0) * 1000
                if action is None:
                    verdict, shown = "PARSE-FAIL", repr(raw)
                else:
                    verdict = "OK" if action == gt else "MISS"
                    shown = f"{ACTION_NAMES[action]} ({action})"
                print(f"  [{'prompt_lm_zero_shot':14s}] answer: {shown}  {verdict}  | {ms:.0f} ms")
                print(f"    raw: {raw!r}")


if __name__ == "__main__":
    main()
