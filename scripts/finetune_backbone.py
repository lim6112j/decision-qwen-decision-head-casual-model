#!/usr/bin/env python
"""Fine-tune the bge-m3 encoder (LoRA) + the dynamic decision head end-to-end
on LocalLLaMA/typed-decisions.

Tests whether adapting the backbone closes the gap to Laya (frozen head 66.3%
-> Laya 76.6%, both on the same 2,000-decision test split).

The frozen-bank pipeline embeds every text once via llama-server and trains the
head on cached vectors. Here the encoder is IN the loop: its LoRA parameters and
the head train jointly, so the text->vector map adapts (Laya-style) — the thing
the frozen GGUF route cannot do.

    HF_HOME=models/.cache/huggingface uv run python scripts/finetune_backbone.py \\
        --train /tmp/td_train.jsonl --test /tmp/td_test.jsonl \\
        --head models/head_dynamic_bge_realonly.pt --epochs 4

Reads the JSONL from scripts/convert_typed_decisions.py.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from decision_lab.head.dynamic_model import (
    DynamicDecisionHead,
    is_correct_dynamic,
    load_dynamic_head,
)
from decision_lab.states.dataset import TextState
from decision_lab.states.fields import state_field_set

MODEL_ID = "BAAI/bge-m3"
TYPE_ID = {"choice": 0, "score": 1, "noul": 2}


def load_items(path: str) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        st = TextState(doc_id=-1, state_type="custom", text=r["text"], labels={})
        q = r["question"]
        kind = q["type"]
        options = q["options"] if kind == "choice" else (
            q["levels"] if kind == "score" else ["false", "true"])
        out.append({
            "fields": state_field_set(st, True), "options": list(options),
            "question": q.get("question") or "", "kind": kind,
            "gold_idx": int(r["gold_idx"]),
        })
    return out


def encode(model, tok, texts: list[str], device, max_len: int = 512) -> torch.Tensor:
    """CLS embedding for each text → (n, D). Grad flows (call under no_grad for eval)."""
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len).to(device)
    return model(**enc).last_hidden_state[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID, help="encoder HF id")
    ap.add_argument("--frozen-encoder", action="store_true",
                    help="freeze the encoder entirely, train the head only "
                         "(the frozen baseline for the same encoder)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--head", default=None, help="warm-start head checkpoint")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2, help="grad accumulation steps")
    ap.add_argument("--max-len", type=int, default=256,
                    help="encoder truncation length (MPS memory scales with it)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-lora", action="store_true", help="full fine-tune the encoder")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast for the encoder (halves MPS activation memory)")
    ap.add_argument("--freeze-below", type=int, default=0,
                    help="train only encoder layers >= this index (detach below; "
                         "0 = all layers). Cuts MPS backward memory.")
    ap.add_argument("--out", default="models/head_dynamic_bge_ft.pt")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model).to(device)
    encoder.config.use_cache = False

    # MPS (20 GB) cannot hold a 568M encoder's full backward. `--freeze-below K`
    # detaches the hidden state after layer K-1, so the backward stops there:
    # only layers K..N-1 are trained, and their activations are the only ones
    # kept. K=0 disables it.
    if args.freeze_below > 0:
        n_layers = len(encoder.encoder.layer)
        boundary = encoder.encoder.layer[args.freeze_below - 1]

        def _detach(_m, _inp, out):
            if isinstance(out, tuple):
                return (out[0].detach(),) + tuple(out[1:])
            return out.detach()

        boundary.register_forward_hook(_detach)
        print(f"partial fine-tune: layers {args.freeze_below}..{n_layers - 1} "
              f"(below detached, saves backward memory)")

    if args.frozen_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        enc_params = []
        print("encoder: FROZEN (head-only baseline for this encoder)")
    elif args.no_lora:
        for p in encoder.parameters():
            p.requires_grad_(True)
        enc_params = list(encoder.parameters())
        print("encoder: FULL fine-tune")
    else:
        from peft import LoraConfig, get_peft_model
        lora_kw = dict(r=args.lora_r, lora_alpha=args.lora_alpha,
                       lora_dropout=0.05, bias="none",
                       target_modules=["query", "key", "value"])
        if args.freeze_below > 0:
            lora_kw["layers_to_transform"] = list(
                range(args.freeze_below, len(encoder.encoder.layer)))
        encoder = get_peft_model(encoder, LoraConfig(**lora_kw))
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        n = sum(p.numel() for p in enc_params)
        print(f"encoder: LoRA r={args.lora_r} a={args.lora_alpha} ({n/1e6:.2f}M trainable)")

    dim = encoder.config.hidden_size
    head = (load_dynamic_head(args.head) if args.head
            else DynamicDecisionHead(input_dim=dim, hidden_dim=256, d_k=128,
                                     dropout=0.1, state_set=True)).to(device)
    head.train()

    def enc(texts: list[str]) -> torch.Tensor:
        """Encode with the encoder; fp16 autocast under --amp to save memory."""
        if args.frozen_encoder:
            with torch.no_grad():          # no backward → much faster
                return encode(encoder, tok, texts, device, args.max_len)
        if args.amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                e = encode(encoder, tok, texts, device, args.max_len)
            return e.float()
        return encode(encoder, tok, texts, device, args.max_len)

    train = load_items(args.train)
    test = load_items(args.test)
    print(f"train {len(train)} | test {len(test)} samples")

    groups = [{"params": head.parameters(), "lr": args.lr_head}]
    if enc_params:
        groups.insert(0, {"params": enc_params, "lr": 2e-5})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = (len(train) + args.batch - 1) // args.batch

    for epoch in range(1, args.epochs + 1):
        random.Random(epoch).shuffle(train)
        head.train()
        t0, tot = time.time(), 0.0
        opt.zero_grad()
        for si, i in enumerate(range(0, len(train), args.batch)):
            batch = train[i : i + args.batch]
            # Flatten every text in the batch into ONE encoder forward.
            flat, spans, n_opt, n_fld = [], [], [], []
            for s in batch:
                start = len(flat)
                flat += s["fields"] + s["options"] + [s["question"]]
                spans.append(start)
                n_fld.append(len(s["fields"]))
                n_opt.append(len(s["options"]))
            with torch.enable_grad():
                embs = enc(flat)  # grad flows into the encoder
            max_m, max_n = max(n_fld), max(n_opt)
            B = len(batch)
            xb = torch.zeros(B, max_m, embs.shape[-1], device=device)
            mask = torch.zeros(B, max_m, dtype=torch.bool, device=device)
            ob = torch.zeros(B, max_n, embs.shape[-1], device=device)
            qb = torch.zeros(B, embs.shape[-1], device=device)
            qtb = torch.empty(B, dtype=torch.long, device=device)
            yb = torch.tensor([s["gold_idx"] for s in batch], device=device)
            for j, s in enumerate(batch):
                o = spans[j] + len(s["fields"])
                xb[j, :n_fld[j]] = embs[o - len(s["fields"]) : o]
                mask[j, :n_fld[j]] = True
                ob[j, :n_opt[j]] = embs[o : o + len(s["options"])]
                qb[j] = embs[o + len(s["options"])]
                qtb[j] = TYPE_ID[s["kind"]]
            logits = head.forward_choice(xb, ob, mask, qb, qtb)
            loss = F.cross_entropy(logits, yb) / args.accum
            loss.backward()
            if (si + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
            tot += loss.item() * args.accum
        print(f"  epoch {epoch}: loss={tot/steps:.4f}  ({time.time()-t0:.0f}s)")

    # ---- eval on the test split (greedy, one sample at a time) ----
    head.eval()
    with torch.no_grad():
        ok = 0
        for s in test:
            flat = s["fields"] + s["options"] + [s["question"]]
            embs = enc(flat)
            o = len(s["fields"])
            xb = embs[:o].unsqueeze(0)
            ob = embs[o : o + len(s["options"])]
            qb = embs[-1].unsqueeze(0)
            qtb = torch.tensor([TYPE_ID[s["kind"]]], device=device)
            pred = head.forward_choice(xb, ob, None, qb, qtb).argmax(dim=-1).item()
            if s["kind"] == "choice":
                gold = s["options"][s["gold_idx"]]
                pred = s["options"][pred]
            elif s["kind"] == "noul":
                gold, pred = bool(s["gold_idx"]), bool(pred)
            else:
                gold = s["gold_idx"]
            ok += is_correct_dynamic({"type": s["kind"]}, pred, gold)
    print(f"\nTEST accuracy: {ok}/{len(test)} = {100*ok/len(test):.1f}%")
    torch.save({"head": head.state_dict(),
                "encoder_lora": {k: v for k, v in encoder.state_dict().items()
                                 if "lora_" in k}},
               args.out)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
