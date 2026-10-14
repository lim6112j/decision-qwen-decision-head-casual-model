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

Budget/speed notes:
- `--precision bf16` autocasts the encoder forward/backward; on A100/H100 TF32
  is enabled by default, which is what makes the fp32-route matmuls fast.
- The train JSONL is split by CASE id (10% val, 10% calibration): the five
  decisions of a case share one state text, so a row-level split would leak.
- Best-val weights are restored before the temperature fit and the test pass.
- `--debug-steps N` stops after N optimizer steps (smoke tests).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from decision_lab.head.dynamic_model import (
    DynamicDecisionHead,
    load_dynamic_head,
)
from decision_lab.states.dataset import TextState
from decision_lab.states.fields import state_field_set

MODEL_ID = "BAAI/bge-m3"
TYPE_ID = {"choice": 0, "score": 1, "noul": 2}
# A temperature fit on a small/short-run calibration slice can run away to a
# huge T (all-confident logits, a handful of rows) and then report garbage
# confidence. Real calibrations on this data land near 1-3, so clamp hard.
T_MIN, T_MAX = 0.10, 20.0


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
            "id": r.get("id", ""),           # case id — the split unit
            "fields": state_field_set(st, True), "options": list(options),
            "question": q.get("question") or "", "kind": kind,
            "gold_idx": int(r["gold_idx"]),
        })
    return out


def split_by_case(
    items: list[dict], val_frac: float, calib_frac: float, seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Group-wise train/val/calibration split over CASES.

    Every decision of a case lands in exactly one split: a case's ~5 decisions
    share the state text, so splitting by row would put near-duplicates on both
    sides of the boundary and inflate val accuracy.
    """
    cases = sorted({it["id"] for it in items})
    random.Random(seed).shuffle(cases)
    n_val = max(1, int(len(cases) * val_frac))
    n_cal = max(1, int(len(cases) * calib_frac))
    val_ids = set(cases[:n_val])
    calib_ids = set(cases[n_val : n_val + n_cal])
    train = [it for it in items if it["id"] not in val_ids and it["id"] not in calib_ids]
    return (train,
            [it for it in items if it["id"] in val_ids],
            [it for it in items if it["id"] in calib_ids])


def ece(confs, hits, n_bins: int = 10) -> float:
    """Expected calibration error over max-prob confidence (equal-width bins)."""
    c = np.asarray(confs, dtype=np.float64)
    h = np.asarray(hits, dtype=np.float64)
    if not len(c):
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi)
        if m.any():
            total += m.mean() * abs(h[m].mean() - c[m].mean())
    return float(total)


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
    ap.add_argument("--batch", type=int, default=32,
                    help="effective batch: samples per optimizer step")
    ap.add_argument("--accum", type=int, default=4,
                    help="gradient accumulation steps; --batch is split into this "
                         "many micro-batches, so only one micro-batch is in flight "
                         "at a time (this is what bounds GPU memory)")
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="physical samples per forward/backward, overriding the "
                         "--accum split; 0 = ceil(--batch / --accum)")
    ap.add_argument("--max-len", type=int, default=256,
                    help="encoder truncation length (MPS memory scales with it)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--no-lora", action="store_true", help="full fine-tune the encoder")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32",
                    help="autocast dtype for the encoder forward/backward. bf16 is "
                         "the A100 choice: fp32's exponent range (no GradScaler, no "
                         "overflow) at tensor-core speed")
    ap.add_argument("--amp", action="store_true",
                    help="deprecated alias for --precision fp16")
    ap.add_argument("--no-tf32", action="store_true",
                    help="disable TF32 matmul/cudnn (enabled by default on CUDA; "
                         "fp32 matmuls run ~8x slower without it on A100)")
    ap.add_argument("--eval-batch", type=int, default=32,
                    help="samples per forward when scoring val/calib/test; the "
                         "encoder sees ~15 texts per sample, so this is a big "
                         "flattened batch (256 OOMs) and must stay small")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of train CASES held out for best-val selection")
    ap.add_argument("--calib-frac", type=float, default=0.1,
                    help="fraction of train CASES held out for the temperature fit")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the val/calib case split and per-epoch shuffles")
    ap.add_argument("--debug-steps", type=int, default=0,
                    help="stop after N optimizer steps (smoke test; 0 = off)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient checkpointing on the encoder: recompute activations "
                         "in the backward instead of storing them. Trades ~30%% compute "
                         "for a large memory cut — needed for a full backward at a big "
                         "batch (no-op with --frozen-encoder, which has no backward)")
    ap.add_argument("--grad-checkpoint-every", type=int, default=1,
                    help="checkpoint every Nth encoder layer (2 halves the recompute "
                         "cost; 1 = every layer)")
    ap.add_argument("--freeze-below", type=int, default=0,
                    help="train only encoder layers >= this index (detach below; "
                         "0 = all layers). Cuts MPS backward memory.")
    ap.add_argument("--out", default="models/head_dynamic_bge_ft.pt")
    args = ap.parse_args()
    if args.amp and args.precision == "fp32":
        args.precision = "fp16"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    if device.type == "cuda" and not args.no_tf32:
        # fp32 matmuls default to full precision; the A100's tensor cores do
        # TF32 at ~8x that rate and nothing here needs fp32 exactness (the head
        # is 0.8M params — its fp32 matmuls are noise either way).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("tf32: enabled (matmul + cudnn)")
    print(f"precision: {args.precision}")
    mem_log = bool(os.environ.get("MEM_LOG"))

    def _mem(tag: str) -> None:
        if mem_log and device.type == "cuda":
            free, _ = torch.cuda.mem_get_info()
            print(f"[mem] {tag:28s} free={free/1e9:6.2f}GB "
                  f"alloc={torch.cuda.memory_allocated()/1e9:6.2f}GB")

    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  {torch.cuda.get_device_name(0)}  "
              f"free={free/1e9:.2f}GB total={total/1e9:.2f}GB  "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    _mem("after device select")
    tok = AutoTokenizer.from_pretrained(args.model)
    _mem("after tokenizer")
    encoder = AutoModel.from_pretrained(args.model).to(device)
    encoder.config.use_cache = False
    _mem("after encoder .to(device)")

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
    _mem("after encoder setup")

    # Gradient checkpointing: recompute layer activations in the backward instead
    # of storing them. The physical micro-batch (x ~15 texts) is what drives
    # memory, so this plus the accumulation split below is what lets a single
    # 40 GB GPU hold batch 32 / len 256. get_peft_model already enabled input
    # grads for the LoRA route; this adds the per-layer checkpointing itself.
    if args.grad_checkpoint and not args.frozen_encoder:
        encoder.gradient_checkpointing_enable(
            every_n_layers=args.grad_checkpoint_every)
        n_ck = sum(1 for m in encoder.modules()
                   if getattr(m, "gradient_checkpointing", False))
        print(f"gradient checkpointing: every {args.grad_checkpoint_every} layer(s) "
              f"({n_ck} modules checkpointed)")

    dim = encoder.config.hidden_size
    head = (load_dynamic_head(args.head) if args.head
            else DynamicDecisionHead(input_dim=dim, hidden_dim=256, d_k=128,
                                     dropout=0.1, state_set=True)).to(device)
    head.train()
    _mem("after head")

    def enc(texts: list[str]) -> torch.Tensor:
        """Encode with the encoder; autocast under --precision fp16/bf16.

        Embeddings come back as fp32 whatever the autocast dtype: the head and
        the cross-entropy stay fp32, so only the encoder's 27 layers run in
        reduced precision (and LoRA/optimizer state stays fp32 untouched).
        """
        if args.frozen_encoder:
            with torch.no_grad():          # no backward → much faster
                return encode(encoder, tok, texts, device, args.max_len)
        if args.precision != "fp32":
            dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
            with torch.autocast(device_type=device.type, dtype=dtype):
                e = encode(encoder, tok, texts, device, args.max_len)
            return e.float()
        return encode(encoder, tok, texts, device, args.max_len)

    train_all = load_items(args.train)
    test = load_items(args.test)
    train, val, calib = split_by_case(train_all, args.val_frac, args.calib_frac, args.seed)
    n_cases = len({i["id"] for i in train_all})
    print(f"train {len(train)} | val {len(val)} | calib {len(calib)} | test {len(test)} "
          f"decisions  ({n_cases} train cases, split by case id)")

    # Gradient accumulation. One micro-batch is in flight at a time, so GPU
    # memory scales with `micro` (the physical forward), not with the effective
    # --batch: the encoder sees ~15 texts per sample, so a full 32-sample
    # forward flattened ~480 sequences and blew past 40 GB. Gradients from
    # `group` micro-batches are summed before each optimizer step, so the update
    # is the same as one micro*group-sized step.
    micro = args.micro_batch or math.ceil(args.batch / max(1, args.accum))
    group = math.ceil(args.batch / micro)
    print(f"batch: effective {args.batch} = {micro} x {group} accumulating "
          f"micro-batches (physical forward = {micro} samples)")

    groups = [{"params": head.parameters(), "lr": args.lr_head}]
    if enc_params:
        groups.insert(0, {"params": enc_params, "lr": 2e-5})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    opt_steps = math.ceil(len(train) / args.batch)

    def logits_for(batch: list[dict]):
        """Flatten a sample batch into ONE encoder forward → (logits, gold)."""
        flat, spans, n_fld, n_opt = [], [], [], []
        for s in batch:
            start = len(flat)
            flat += s["fields"] + s["options"] + [s["question"]]
            spans.append(start)
            n_fld.append(len(s["fields"]))
            n_opt.append(len(s["options"]))
        embs = enc(flat)  # grad flows into the encoder while training
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
        return head.forward_choice(xb, ob, mask, qb, qtb), yb

    def snapshot() -> dict:
        """Best-val weights on CPU; encoder saves its LoRA tensors only (2.4M)."""
        return {
            "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            "encoder_lora": {k: v.detach().cpu().clone()
                             for k, v in encoder.state_dict().items() if "lora_" in k},
        }

    def evaluate(items: list[dict]) -> tuple[float, list, list]:
        """Greedy accuracy + (confidence, correct) pairs for ECE/calibration."""
        head.eval()
        encoder.eval()
        n_ok, confs, hits = 0, [], []
        eb = min(micro, args.eval_batch)
        with torch.no_grad():
            for i in range(0, len(items), eb):
                logits, yb = logits_for(items[i : i + eb])
                probs = F.softmax(logits.float(), dim=-1)
                ok = probs.argmax(dim=-1) == yb
                n_ok += int(ok.sum().item())
                confs += probs.max(dim=-1).values.tolist()
                hits += ok.tolist()
        return n_ok / max(1, len(items)), confs, hits

    best_acc, best_epoch, best_snap = -1.0, 0, None
    steps = 0
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train)
        head.train()
        # from_pretrained() returns the encoder in eval() mode. Gradient
        # checkpointing and LoRA dropout only engage in train() mode, so without
        # this the full activation graph is kept and a 114x256 micro-batch needs
        # ~36 GB instead of ~5 GB.
        encoder.train()
        t0, tot, n_micro, in_group = time.time(), 0.0, 0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad()
        step_t0 = time.time()
        for si, i in enumerate(range(0, len(train), micro)):
            batch = train[i : i + micro]
            # Flatten every text in the batch into ONE encoder forward.
            with torch.enable_grad():
                logits, yb = logits_for(batch)
                if si == 0:
                    _mem(f"micro0 post-fwd samples={len(batch)}")
                loss = F.cross_entropy(logits, yb)
            # Normalize by the group so the accumulated gradient is the mean
            # over the effective batch, not the sum over micro-batches.
            (loss / group).backward()
            tot += loss.item()
            n_micro += 1
            in_group += 1
            if in_group == group or i + micro >= len(train):
                opt.step()          # flush the tail group too, don't drop it
                opt.zero_grad()
                in_group = 0
                steps += 1
                if args.debug_steps:
                    print(f"    step {steps}: {time.time() - step_t0:.2f}s "
                          f"(micro={micro} x {group})")
                    step_t0 = time.time()
                    if steps >= args.debug_steps:
                        break
        mem = ""
        if device.type == "cuda":
            mem = (f"  peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB"
                   f" reserved={torch.cuda.max_memory_reserved()/1e9:.1f}GB")
        # Best-val selection: the epoch that generalizes best is the one that
        # ships, not the last one. The val cases are never trained on.
        val_acc, _, _ = evaluate(val)
        marker = ""
        if val_acc > best_acc:
            best_acc, best_epoch, best_snap = val_acc, epoch, snapshot()
            marker = "  *best*"
        print(f"  epoch {epoch}: loss={tot/max(1, n_micro):.4f}  val_acc={val_acc:.4f}  "
              f"({opt_steps} opt steps, {time.time()-t0:.0f}s){mem}{marker}")
        if args.debug_steps and steps >= args.debug_steps:
            print(f"  debug: stopped after {steps} optimizer steps")
            break

    if best_snap is not None:
        head.load_state_dict(best_snap["head"])
        if best_snap["encoder_lora"]:
            encoder.load_state_dict(best_snap["encoder_lora"], strict=False)
        print(f"  restored best weights: epoch {best_epoch}, val_acc={best_acc:.4f}")

    # ---- metrics: accuracy, NLL and ECE before/after temperature scaling ----
    # Temperature only rescales the softmax, so accuracy is T-invariant; NLL
    # and ECE are what the calibration is for.
    head.eval()
    encoder.eval()

    def clamp_t(t: float) -> float:
        return min(max(float(t), T_MIN), T_MAX)

    def logit_batches(items: list[dict], bs: int = 0):
        bs = bs or args.eval_batch
        out = []
        with torch.no_grad():
            for i in range(0, len(items), bs):
                logits, yb = logits_for(items[i : i + bs])
                out.append((logits.float().cpu(), yb.cpu(), len(yb)))
        return out

    def _nll(batches, t: float) -> float:
        """Mean NLL of temperature-scaled logits, clamped at 0.

        cross_entropy() accepts negative targets (NLL = -log p), so a P(0)
        underflow or a huge T can drive the mean below 0, which is not a
        possible NLL — clamp rather than report it.
        """
        total = sum(n for _, _, n in batches)
        if not total:
            return float("nan")
        with torch.no_grad():
            s = sum(float(F.cross_entropy(lg / clamp_t(t), lb, reduction="sum"))
                    for lg, lb, _ in batches)
        return max(0.0, s / total)

    def _confs(batches, t: float) -> list:
        return [c for lg, _, _ in batches
                for c in F.softmax(lg / clamp_t(t), -1).max(-1).values.tolist()]

    def report(name: str, items: list[dict]) -> dict:
        batches = logit_batches(items)
        hits = [bool(h) for lg, lb, _ in batches for h in (lg.argmax(-1) == lb).tolist()]
        acc = sum(hits) / max(1, len(hits))
        m = {"n": len(items), "acc": acc,
             "nll": _nll(batches, 1.0), "nll_calibrated": _nll(batches, temperature),
             "ece": ece(_confs(batches, 1.0), hits),
             "ece_calibrated": ece(_confs(batches, temperature), hits)}
        print(f"  {name:5s} n={m['n']:5d}  acc={acc:.4f}  "
              f"NLL {m['nll']:.4f}→{m['nll_calibrated']:.4f}  "
              f"ECE {m['ece']:.4f}→{m['ece_calibrated']:.4f}")
        return m

    # ---- temperature calibration on the calibration cases ----
    # One global scalar: the question types mix option counts (2 options vs 4
    # levels), so the logits are ragged — the closure sums batch-wise weighted
    # CE instead of concatenating them. Fit on the calibration slice, never on
    # the val slice that picked the checkpoint.
    calib_batches = logit_batches(calib)
    if calib_batches:
        log_t = torch.zeros(1, requires_grad=True)
        cal_opt = torch.optim.LBFGS([log_t], lr=1e-2, max_iter=1000)
        total_n = sum(n for _, _, n in calib_batches)

        def closure():
            cal_opt.zero_grad()
            t = log_t.exp().clamp_min(1e-3)
            loss = sum(F.cross_entropy(lg / t, lb, reduction="sum") / total_n
                       for lg, lb, _ in calib_batches)
            loss.backward()
            return loss

        cal_opt.step(closure)
        temperature = clamp_t(log_t.exp().item())
    else:
        temperature = 1.0
    print(f"\ncalibration: T={temperature:.3f}  "
          f"NLL {_nll(calib_batches, 1.0):.4f} → {_nll(calib_batches, temperature):.4f}")

    metrics = {"temperature": temperature, "best_epoch": best_epoch,
               "val_acc_best": best_acc, "args": vars(args),
               "val": report("val", val), "test": report("test", test)}
    print(f"\nTEST accuracy: {metrics['test']['acc']*100:.1f}%  "
          f"({best_epoch=}, overall best val_acc={best_acc:.4f})")

    torch.save({"head": head.state_dict(),
                "encoder_lora": {k: v for k, v in encoder.state_dict().items()
                                 if "lora_" in k},
                "temperature": temperature,
                "metrics": metrics},
               args.out)
    metrics_path = Path(args.out).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved → {args.out}  (+ {metrics_path})")


if __name__ == "__main__":
    main()
