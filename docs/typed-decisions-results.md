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
4. **The remaining gap to Laya is ~10pp** (66.3% vs 76.6%). Candidate causes:
   (a) Laya fine-tunes its backbone end-to-end (we freeze ours) — **now tested:
   worth only +2.9pp, so not the main cause** (see "Backbone adaptation"
   below); (b) Laya encodes state+question in one sequence (we embed them
   separately); (c) head-design differences.
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

## Backbone adaptation — LoRA is worth +2.9pp, a third of the gap (2026-10-14)

`scripts/finetune_backbone.py` trains the bge-m3 encoder (LoRA r=16 + the
~820k head, end-to-end) — the Laya-style route the frozen llama-server pipeline
cannot take. First attempted on MPS and parked (2026-10-13): a full backward
through the 568M encoder needed `--freeze-below` + fp16, at 80 min–4.5 h per
epoch. **Now run on one A100-40GB.**

### Headline: 66.3% → 69.2% — real, but only a third of the way to Laya

| setting | typed-decisions test | backbone | train time |
|---|---|---|---|
| frozen bge-m3, real-only (2026-10-13) | 66.3% | frozen | 52 s |
| adapted, **4 epochs** (2026-10-13, fp32) | 59.5% | LoRA | 3.4 h |
| **adapted, 15 epochs** (2026-10-14, best-val @ ep 12) | **69.2%** | LoRA | **2.1 h** |
| Laya (fine-tuned ModernBERT), model card | 76.6% | fine-tuned | ~5 min |

**Adapting the backbone is worth +2.9pp over freezing it** (66.3 → 69.2), not
the ~10pp the gap needed. So freezing the encoder was **not** the main cause of
the gap to Laya; the remaining ~7pp is still open (their ModernBERT, one
sequence per sample, head design, 6 epochs on 6,000 decisions vs our 12).

### Why 4 epochs was worse than doing nothing (59.5%)

The first run's 4 epochs were **compute-starved, not hyperparameter-limited**:
fp32 with **TF32 matmul off** (`torch.backends.cuda.matmul.allow_tf32` defaults
to False; `float32_matmul_precision` was `highest`), so each epoch cost 50 min.
The run looked like “adaptation hurts” when it was just under-trained. Two
fixes: enable TF32 + autocast **bf16** on the encoder (embeddings and the head
stay fp32), and cap eval forwards (`--eval-batch`; 256 samples flatten ~3.8k
texts and OOM). 50 min/epoch → **8.2 min/epoch**, ~6x.

### The 15-epoch curve (per-epoch val, 600 decisions / 120 cases)

Loss falls monotonically, val peaks at **epoch 12** (0.7033) and is flat-to-down
after: ep 9 0.6767, ep 12 **0.7033**, ep 15 0.6900. Best-val selection is worth
0.13pp here — small, but the last epoch is demonstrably not the best one.

### Calibration (temperature, fit on 600 held-out decisions never trained on)

| split | acc | NLL | ECE |
|---|---|---|---|
| calibration, T=1 | – | 0.8564 | – |
| calibration, T=2.193 | – | **0.7104** | – |
| val (600) | 0.7033 | 0.9608 → **0.7518** | 0.1246 → **0.0708** |
| **test (2000)** | **0.6925** | 0.8223 → **0.6905** | 0.1207 → **0.0361** |

Temperature scaling **cuts ECE 3.3x** (0.1207 → 0.0361) and NLL 16%, with
accuracy untouched (softmax rescaling is rank-preserving). The model was
over-confident — it needed T>1. **T is clamped to [0.1, 20]**: an unbounded
LBFGS fit on a small or short-run calibration slice runs away to T in the
thousands and reports meaningless confidence (observed: T=3186 on a 5-row
slice).

### Where it helps — the gain is not uniform

By question type: `choice` **50.7% → 68.0% (+17.3pp)**, `score` +8.5pp,
`noul` +4.0pp. The frozen backbone's weakness was exactly the hardest, most
compositional questions — `category` **+43pp** (51→94%), `churn_risk` +23pp,
`discrepancy_severity` +21pp, `outcome` +19pp. Two questions moved *backwards*
(`risk` −10pp, `matches_order` −1pp); `duplicate` (+0.0pp) was already at 89%
and stayed. By workflow, customer service gained most (+20.2pp), the three
others +5–7pp.

### Reproduce

```bash
# one-off: backbone adaptation on a GPU, ~2 h
CUDA_VISIBLE_DEVICES=1 uv run python scripts/finetune_backbone.py \
  --model BAAI/bge-m3 --train /tmp/td_train.jsonl --test /tmp/td_test.jsonl \
  --epochs 15 --batch 32 --micro-batch 16 --eval-batch 32 --max-len 256 \
  --precision bf16 --grad-checkpoint --seed 42 \
  --out models/head_dynamic_bge_ft15.pt
```

`--grad-checkpoint` is what keeps the 16-sample micro-batch (≈230 texts) at
14.2 GB; without it bf16 alone OOMs at that micro-batch. `CUDA_VISIBLE_DEVICES`
remaps the visible GPU to `cuda:0`, so "GPU 0" in the logs/errors is physical
GPU 1. Metrics land in `models/head_dynamic_bge_ft15.metrics.json`; the
checkpoint carries `head`, `encoder_lora`, `temperature` and `metrics`.

### Caveats

- **The checkpoint's `encoder_lora` needs the PEFT-wrapped bge-m3 to load**, so
  it is **not** drop-in for the llama-server serving path (that path takes a
  GGUF and a head-only checkpoint). Serving the adapted model means either
  merging the LoRA into the encoder or a non-GGUF server.
- Test accuracy is computed here as the argmax over `forward_choice` logits;
  69.2% (this script) vs 69.3% (its own pass) is the sampling difference
  between two eval loops, not a real change.
- Single seed (42) on a single split. The synthetic-era held-out spread was
  ~8pp, so treat ±2pp as noise and anything under ~5pp as unresolved.

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
- The command above is the **MPS-era recipe** (`--freeze-below`/`--amp` are
  MPS workarounds). On a GPU use the bf16 recipe in the section above.
- `--frozen-encoder` runs the **frozen baseline for the same encoder**
  (head-only), so you can A/B adapt-vs-frozen on bge-m3 itself.
- The `66.3%` frozen baseline in the table above was produced by the **local
  llama-server pipeline** (`build-real` + `train-dynamic` real_only), which
  needs `models/bge-m3-Q8_0.gguf` + llama.cpp — both gitignored, so reproduce
  those numbers only if you also fetch the GGUF. The `finetune_backbone.py`
  route needs neither.

## Next

1. **Close the remaining ~7pp to Laya** (69.2% vs 76.6%). Frozen-vs-adapted is
   now measured and worth only +2.9pp, so the next suspects are the ones left
   untested: one sequence per sample, a `ModernBert`/long-context encoder, and
   the 12-vs-6 epoch schedule.
2. **Merge the LoRA into the encoder** so the adapted model can actually be
   served — today the checkpoint only loads through PEFT (see caveats).
3. Test **joint state+question encoding** (one sequence) vs our separate
   embeddings — cheap, still open.
4. Adopt this real-only + typed-decisions setup as the standard experiment
   harness (52 s per run frozen; 2.1 h adapted).
