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

## Next

1. Test whether **backbone adaptation** (LoRA / full FT on bge, Laya-style)
   closes the ~10pp — needs a PyTorch backbone path (llama-server is
   forward-only GGUF).
2. Test **joint state+question encoding** (one sequence) vs our separate
   embeddings.
3. Adopt this real-only + typed-decisions setup as the standard experiment
   harness (52 s per run).
