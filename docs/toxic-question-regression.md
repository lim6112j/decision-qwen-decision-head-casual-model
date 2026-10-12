# Parked: `toxic` (noul) question — works, but regresses `quality`

Recorded 2026-10-12. **Status: parked, not shipped.** The `toxic` question was
added to the dynamic head and trained successfully, but the same training run
collapsed the `quality` rubric on held-out. The code and data are preserved so
the work can be resumed; the served checkpoint is back on the pre-`toxic`
baseline.

## Why this exists

The served dynamic head had no toxicity concept. The exact case

```json
{"state": "You are absolutely worthless and everyone here hates you. Do "
          "everyone a favor and delete your account, you pathetic loser.",
 "question": "toxic", "option": "true/false"}
```

returned `false` at confidence **0.538** — a coin flip — with a distribution
identical (<0.001) for the bare qid `"toxic"` and the natural phrasing
`"Is this text toxic?"`. Trained questions condition strongly and the two
phrasings differ; this inertness is the signature of an untrained question.

## What was built

- **Bank entry** — `toxic` + `not_toxic` in `src/decision_lab/head/question_bank.py`
  (`DOC_CORES`, `doc_specs`, and the `DOC_LABEL_GATED` label-key gate). Gated so
  it is applicable only to states that actually carry a `toxic` gold; a bare
  `_is_doc` gate would `KeyError` in `generate_dynamic_training_data`'s `emit()`
  loop, because the synthetic generator emits no `toxic` label.
- **Gold from real labels** — the synthetic generator has no `toxic` latent, so
  the training signal comes entirely from the real-traffic pipeline.
  `scripts/build_toxic_labels.py` authors a curated seed corpus (55 toxic /
  38 benign / 18 borderline = 111 states) → `data_bge/real/labels.jsonl`
  (1,332 rows across 7 `toxic` + 5 `not_toxic` phrasings) → `build-real` →
  `data_bge/real/samples.pt` (1,066 train / 266 held-out).
- **Config** — `dynamic_head.real_data_path: "data_bge/real/samples.pt"`,
  `real_weight: 3` (real samples enter **stage 2**, `is_variant=True`).
- **Tests** — `tests/test_question_bank.py`: canonical texts, gold, and the
  applicability gate. Full suite: 276 passed.

## Result 1 — `toxic` works

Retrained head (`models/head_dynamic_bge_toxic.pt`, val_acc 0.974, T=1.001):

| state | `"toxic"` | `"Is this text toxic?"` | paraphrase | `"Is this text NOT toxic?"` |
|---|---|---|---|---|
| abusive post | **true** 1.000 | **true** 1.000 | **true** 1.000 | false 1.000 |
| benign invoice | false 1.000 | false 1.000 | false 1.000 | true 0.991 |
| borderline (harsh, not abusive) | false 1.000 | false 1.000 | false 1.000 | true 0.584 |

Held-out real set: 266/266 — but that is held-out over the *curated seed
corpus* and is nearly meaningless as a generalization estimate. The head needs
out-of-distribution toxic text to be trustworthy.

## Result 2 — `quality` regressed (the blocker)

A/B against the preserved pre-change checkpoint
(`models/head_dynamic_bge_pre_toxic.pt`, which reproduces the README's numbers
exactly — 91.6% held-out):

| question (held-out) | pre-toxic | post-toxic |
|---|---|---|
| sentiment / urgency | 1.000 / 1.000 | 1.000 / 1.000 |
| **quality** (4-level rubric) | **0.934** | **0.527** |
| is_actionable | 0.979 | 0.943 |
| contains_pii | 0.777 | 0.953 |
| is_urgent | 0.817 | 0.879 |
| **overall** | 0.916 | 0.882 |

Only `quality` broke — the question the README flags as *most dependent on
reading the whole document*.

## Hypothesis (unconfirmed)

The head is a single shared trunk; questions enter as embeddings, so there is
no per-question parameter set to train in isolation. The 3,198 real samples
(1,066 × weight 3) are short, few-field, `noul` samples that enter **stage 2**
and carry high loss every epoch while `toxic` is being learned, biasing the
trunk away from the multi-field reasoning `quality` needs. `toxic` and
`quality` fight over the same trunk.

Not yet isolated: whether the culprit is the real mix or something else in the
change. The clean experiment is to retrain with the real mix OFF (bank change
kept) — if `quality` returns to ~93%, the real samples are confirmed.

## Reproduce

```bash
python scripts/build_toxic_labels.py                # data_bge/real/labels.jsonl
# set dynamic_head.real_data_path: "data_bge/real/samples.pt", real_weight: 3
no_proxy=localhost,127.0.0.1 uv run python -m decision_lab \
    --config configs/bge_full.yaml --data-dir data_bge build-real
no_proxy=localhost,127.0.0.1 uv run python -m decision_lab \
    --config configs/bge_full.yaml --data-dir data_bge train-dynamic
```

A `train-dynamic` run is ~70 min: ~20 min re-embedding option/question/shape
texts live (only state field features are cached) + ~50 min gradients, over
the whole bank — independent of how many questions changed.

Artifacts (gitignored, present locally): `models/head_dynamic_bge_toxic.pt`
(the regressed head), `models/head_dynamic_bge_pre_toxic.pt` (baseline),
`data_bge/real/{labels.jsonl,samples.pt,test_real.jsonl}`.

## If resumed

1. Isolate the cause (retrain with the real mix off).
2. Candidate fixes: lower `real_weight`; add the real samples only in a short
   final fine-tune phase at low LR; or a dedicated sub-head for `toxic` so it
   is not learned on the shared trunk.
3. Also revisit the corpus: the seed is small and easily separable — collect
   real out-of-distribution toxic text before trusting the numbers.
