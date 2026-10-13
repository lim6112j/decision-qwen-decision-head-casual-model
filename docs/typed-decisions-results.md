# typed-decisions: the real-data pivot (2026-10-13)

Recorded 2026-10-13. Supersedes the synthetic-bank direction. The synthetic
bank is **deprecated as a training source** — it is inflated (see below) and
contributes ~nothing on real data.

## Why we pivoted

The synthetic bank (344k samples from ~10k templated states) turned out to be
mostly redundant: the document latents span only `3 sentiments x 4 urgencies x
4 qualities x 2 actionable x 2 pii = 192` label combinations, so 10k states are
~52 re-renderings of each; ~half the samples were shape-augmentation
duplicates. Shrinking to 40% of the samples (136k) trained **4x faster with no
accuracy loss**, and training instability was large (seed-only held-out spread
**~8pp**: 0.916 vs 0.993 for the same config). See
[toxic-question-regression.md](toxic-question-regression.md) for the full
synthetic-era trail.

## The dataset

[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
— 1,200 train cases / 6,000 decisions, 400 test cases / 2,000 decisions, four
workflows (agent-trace observability, customer service, invoice processing,
security incidents). Their decisions are the **same typed system** we use:
`choice` (criteria dict), `score` (criteria list), `noul`. This is the data
`convaiinnovations/laya-typed-decisions` fine-tunes on and reports **0.766** on
the test split.

Converted with `scripts/convert_typed_decisions.py` (parquet → our JSONL +
`labels.jsonl`) and evaluated with `scripts/eval_typed_decisions.py`. The case
`state` is a nested JSON object — used verbatim (our field splitter chunks it).

**Gotcha:** their JSON states exceed llama.cpp's default physical batch (512
tokens → HTTP 500 "input is too large to process"). `model.batch_size` was
added to the config and LlamaServer to raise `--batch-size`/`--ubatch-size`.

## Results (our dynamic head, frozen bge-m3, ~820k head)

| setting | typed-decisions test | train time | our bank (held-out) |
|---|---|---|---|
| our head, **no training** on this data | 28.2% (= chance) | – | 0.916 |
| our bank **+ 10%** typed-decisions mixed | **67.2%** | 20.8 min | **0.765** (collapsed) |
| **real-only** (typed-decisions only, bank skipped) | **66.3%** | **52 s** | n/a |
| Laya (fine-tuned ModernBERT), from the model card | **76.6%** | ~5 min | n/a |

### Findings

1. **We can learn this domain.** 28% → 66–67% with a frozen backbone and an
   ~820k head.
2. **The synthetic bank is useless here.** Mixed (67.2%) ≈ real-only (66.3%) —
   90% synthetic data bought ~1pp and cost our bank 0.15.
3. **Multi-domain interference is real and reproducible.** Mixing the new
   domain collapsed our bank 0.916 → 0.765 (every question down). This is the
   `toxic` regression amplified — so that regression was **not** just seed
   noise; the shared frozen head genuinely cannot hold two heterogeneous
   domains.
4. **The remaining gap to Laya is ~10pp** (66.3% vs 76.6%). Candidate causes,
   untested: (a) Laya fine-tunes its backbone end-to-end (we freeze ours);
   (b) Laya encodes state+question in one sequence (we embed them separately);
   (c) head-design differences.
5. **Real-only training is 52 s** (vs 18–70 min on the synthetic bank) — the
   synthetic bank was the iteration bottleneck, not the head.

## Decision

- **Abandon the synthetic bank as a training source.** The default path is
  real-only on the typed-decisions data (`dynamic_head.real_only: true` +
  `real_data_path`). The bank code (`states/generator.py`,
  `head/question_bank.py`) is left in place but is vestigial.
- Keep the synthetic-era checkpoints/experiments for reference only.

## Caveats

- **Not apples-to-apples with Laya**: they trained on this data only and
  fine-tuned the backbone; our 66.3% is a frozen backbone and separate
  embeddings. The ~10pp is a gap to explain, not a like-for-like defeat.
- The 0.766 is the model card's number on the official split, not re-measured
  here.

## Backbone adaptation — attempted, blocked on local hardware (2026-10-13)

`scripts/finetune_backbone.py` trains the bge-m3 encoder (LoRA) + the head
end-to-end on this data — the Laya-style route the frozen llama-server pipeline
cannot take. It **runs**, but is **impractical on this machine (MPS)**:

- **Memory.** A full backward through the 568M encoder does not fit MPS's
  ~20 GB cap. It required `--freeze-below K` (detach the hidden state after
  layer K-1 so only the top layers backprop) plus fp16 autocast. Full fine-tune
  OOMs at any usable batch.
- **Speed.** The typed-decisions JSON states split into **~10 fields** →
  **~15 texts per sample**. Encoding that every step dominates: measured
  **~1 s/sample** (batch 1, len 128) to **~3.4 s/sample** (batch 8, len 256) —
  i.e. **80 min to 4.5 h per epoch** over 4,800 samples.
- **Why Laya's 5 min is not comparable:** 2×T4 GPUs, a 395M model, 6,000
  decisions, and one sequence per sample — not MPS with 15 texts/sample.

**Decision: parked.** The ~10pp gap to Laya (66.3% frozen vs 76.6%) remains
unexplained; testing it needs a GPU (or a much smaller encoder + reduced
fields). The script is left in place, documented as GPU-required.

## Running on a GPU server (fresh clone)

The backbone-adaptation experiment is self-contained (only HF downloads needed):

```bash
pip install -e ".[dev]" && pip install transformers peft pyarrow

# 1. data: parquet → our JSONL (+ labels.jsonl)
mkdir -p /tmp/typed-decisions
cd /tmp/typed-decisions
for s in train test; do
  curl -sL "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/refs%2Fconvert%2Fparquet/all/$s/0000.parquet" -o "all_$s.parquet"
done
cd -        # back to the repo
python scripts/convert_typed_decisions.py /tmp/typed-decisions/all_train.parquet \
    --labels-out data_bge/real/labels.jsonl -o /tmp/td_train.jsonl
python scripts/convert_typed_decisions.py /tmp/typed-decisions/all_test.parquet \
    -o /tmp/td_test.jsonl

# 2. backbone adaptation (GPU): LoRA bge-m3 encoder + head, end-to-end
python scripts/finetune_backbone.py \
    --model BAAI/bge-m3 --train /tmp/td_train.jsonl --test /tmp/td_test.jsonl \
    --epochs 4 --batch 32 --max-len 256      # drop --freeze-below/--amp on a big GPU
```

Notes:
- **On a GPU, drop `--freeze-below`** (the full backward fits in VRAM) and
  `--amp` if you have headroom; a larger batch is much faster. The script picks
  cuda > mps > cpu.
- `--frozen-encoder` runs the **frozen baseline for the same encoder**
  (head-only), so you can A/B adapt-vs-frozen on bge-m3 itself.
- The `66.3%` frozen baseline in the table above was produced by the **local
  llama-server pipeline** (`build-real` + `train-dynamic` real_only), which
  needs `models/bge-m3-Q8_0.gguf` + llama.cpp — both gitignored, so reproduce
  those numbers only if you also fetch the GGUF. The `finetune_backbone.py`
  route needs neither.

## Next

1. Backbone adaptation (above) — needs a GPU.
2. Test **joint state+question encoding** (one sequence) vs our separate
   embeddings — cheap, still open.
3. Adopt this real-only + typed-decisions setup as the standard experiment
   harness (52 s per run).
