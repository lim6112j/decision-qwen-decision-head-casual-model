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

## CRITICAL FINDING — the head is highly seed-unstable (2026-10-13)

V1: retraining the **exact baseline config** (v4 / hidden_dim 256 / no toxic
data / expert separation off) with only the torch RNG (init + shuffling)
differing — the synthetic samples are deterministic (`Random(42)`) and the
architecture is identical — gave held-out **0.993** vs the baseline's
**0.916**. That is **~7.7pp held-out variance from the random seed alone.**

Per-question the weak questions swing even more: `contains_pii` 0.777↔0.995,
`is_urgent` 0.817↔1.000, `quality` 0.527↔0.993 across runs.

**Consequence:** every per-run attribution in this document — "toxic breaks
`quality`", "v5 breaks `is_actionable`", "B degrades the general head",
"512 recovers `quality`" — is within this noise floor and cannot be trusted
from a single run. The real problem is not that `toxic` breaks a specific
question; it is that **this head does not train stably**. Any further
architecture experiment is uninterpretable until the noise floor is known
(retrain the baseline across several seeds) or reduced (seed ensembling /
checkpoint averaging / stronger regularization).

Also confirmed: the 200-state benchmark subsample is *representative* (full
1000-state eval matched it: 0.882 vs 0.884 held-out, per-question within 1.5pp),
and a **full-set eval takes ~28 min** (`benchmark.max_test_states: 0`). So the
variance is model-level, not measurement noise.

## Open ideas (TODO — not yet tested)

- **Adapt the backbone (partial fine-tune / LoRA).** Laya fine-tunes its
  ModernBERT-large encoder end-to-end (encoder LR 2.5e-5, head 1e-4, no LoRA);
  Kev-0.8B uses a LoRA adapter on a frozen Qwen. This repo **freezes** bge-m3
  entirely (llama-server is forward-only GGUF — no gradient path), so the head
  must fit a decision boundary in a *fixed* representation. That is a plausible
  driver of the seed instability above (fixed geometry ⇒ the boundary lands
  differently per init), and of the weak questions (`contains_pii`,
  `is_urgent`) that the docs call "label ambiguity / geometry". The repo's
  `docs/kev-0.8b-comparison.md` argued bge makes LoRA redundant — but that
  argument was about *accuracy*, not *stability*. Cost: adopting this means
  training the encoder in PyTorch (not the frozen-GGUF pipeline) — a big infra
  change. Candidate, not yet attempted.

- **v5 type conditioning — tested, did NOT fix it (2026-10-12).** Feeding the
  answer format (choice/score/noul) as an explicit head input (`type_emb` on
  the question vector + `type_q_shift` on the option queries, arch_version 5,
  `head_dynamic_bge_v5_toxic.pt`) left the overall held-out at 0.888 (vs the
  0.882 v4-toxic and 0.916 no-toxic baseline) — it only *moved* the damage
  (`quality` 0.527→0.807, but `is_actionable` 0.979→0.706). The failure shifts
  between runs, which points at **capacity/stability**, not a routing clash.
- **Isolation experiment A — v5 code is the culprit, not the toxic data
  (2026-10-12).** Training v5 with the real mix **OFF** (no toxic samples at
  all) still gave held-out **0.889** — the same as v5+toxic (0.888) and below
  the 0.916 baseline. Per-label: `is_actionable` 0.979→0.604, `quality`
  level-0 0.83→0.30. So the regression tracks the **v5 code**, not the data.
  Likely cause: `type_emb` was added to the *question vector*, which is the
  field-selection query — the type signal contaminates the content query.
  Preserved as `models/head_dynamic_bge_v5_noreal.pt`; code in
  `git stash@{0}`. Retry **F** = inject the type only at the readout/output,
  never into the field-selection query. Side note: under v5 the toxic mix cost
  ~0 (0.888 ≈ 0.889) — type conditioning did help *robustness to the mix*;
  it just lowered the baseline more than that was worth.
- **Retry F — type at the READ-OUT only — partial (2026-10-12).** Moving the
  type injection off the question vector (which drives field selection) and
  into the read-out (`qz_readout([z; z_q; type_vec])`, 2H→3H input; no
  `type_q_shift`) **fixed `quality` completely: 0.807 → 0.989** (above the
  0.934 baseline) — confirming the field-selection-contamination hypothesis
  *for `quality`*. But `is_actionable` stayed broken (**0.673**; v4 baseline
  0.979) with no class-prior shift (gold-False 0.77, gold-True 0.57 — both leak).
  Overall held-out **0.898** — the best of the modified runs (v4+toxic 0.882,
  v5+toxic 0.888, A 0.889) but still 1.8pp below baseline. `toxic` still works
  (abusive→true 1.000). Preserved as `models/head_dynamic_bge_v5F.pt`; the F
  code is the current working tree.

  **Cross-run pattern (5 runs):** *any* perturbation breaks exactly one
  content-dependent question — v4+toxic breaks `quality` (0.527); every
  type-conditioned run breaks `is_actionable` (~0.60–0.71), regardless of
  injection point. Different runs break different questions ⇒ **the 800k-param
  head is at capacity**; the lever is capacity (raise `hidden_dim` /
  type-specific sub-heads), not more type-injection tweaks.
- **Capacity test C1 — v4 mechanics + toxic data, `hidden_dim` 256→512
  (2026-10-12).** Total recovered to **0.912** (vs 0.882 at 256; baseline
  0.916) and `quality` went **0.527 → 1.000**. But `is_actionable` **collapsed
  to 0.500** (confusion ≈ random both ways, not a class prior) — the damage
  moved again. `toxic` works. Preserved as `models/head_dynamic_bge_cap512.pt`.
- **Six-run summary — the total is stable, the per-question split is not:**

  | run | code | hidden | toxic | heldout | quality | is_actionable |
  |---|---|---|---|---|---|---|
  | pre | v4 | 256 | – | 0.916 | 0.934 | 0.979 |
  | v4+toxic | v4 | 256 | ✓ | 0.882 | 0.527 | 0.943 |
  | v5+toxic | v5-in | 256 | ✓ | 0.888 | 0.807 | 0.706 |
  | A | v5-in | 256 | – | 0.889 | 0.788 | 0.604 |
  | F | v5-readout | 256 | ✓ | 0.898 | 0.989 | 0.673 |
  | C1 | v4 | 512 | ✓ | 0.912 | 1.000 | 0.500 |

  Held-out total stays in 0.88–0.92 across every config, but `quality` swings
  0.527–1.000 and `is_actionable` swings 0.500–0.979. `is_actionable` is high
  (0.94–0.98) only in the two *baseline-shaped* 256/v4 runs and collapses
  under **any** deviation (more capacity, or type conditioning, or the toxic
  mix). This looks like a **fragile/degenerate `is_actionable`** plus a
  capacity-limited shared head, not a routing problem. Next lever with a real
  chance of *no* collateral: **a dedicated sub-head for `toxic`** (capacity
  separation) so the doc questions are never perturbed.
- **Frozen-backbone verbalizer / claim-plausibility (idea, untested).** Idea:
  skip the trained head entirely — turn each option into a natural-language
  claim about the state (e.g. "This text is toxic." vs "This text is not
  toxic.") and let the *frozen* backbone pick the one that "reads well". Two
  concerns to test cheaply: (a) an embedding model (bge) gives *similarity*,
  not *plausibility* — cosine to the context is a weak proxy; (b) negation
  asymmetry — "toxic" and "not toxic" share nearly all content words, so their
  embeddings are near-identical and cosine may not separate them (the bank
  already routes polarity on the *question* text, not option vocabulary, for
  this reason). Also note the repo already has a zero-shot prompt-LM agent
  (`prompt_lm_zero_shot`) whose benchmark numbers are the generative-LM version
  of this question.
- **Direct question↔option bilinear term (idea, untested).** v4 already has the
  question attend the state (`z_q`) and a question×option interaction
  (`qz_readout([z_j ; z_q])`); the delta would be a zero-init `cos(opt_proj(o_j),
  z_q)` term added to the logits (strict extension of v4/v5). Would test whether
  making the option↔context comparison *direct* (pointer-head style, like
  Jev/Laya) helps — but the capacity evidence above makes this lower priority
  than capacity separation.
