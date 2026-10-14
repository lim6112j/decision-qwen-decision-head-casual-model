# The systematic grid dataset (replaces the random synthetic bank)

Recorded 2026-10-14 on branch `v3-bge`. The random synthetic bank is **too big
and too redundant**; a full grid over its latent space is smaller, ~10x faster
to run end to end, and *more* accurate.

## Why the random bank is redundant

`train.jsonl` (10,000 states) measured 2026-10-14:

| metric | value |
|---|---|
| distinct texts (exact duplicates) | 8,838 (**1,162 dups, 11.6%**) |
| **distinct label vectors** | **192** (`3 sentiment x 4 urgency x 4 quality x 2 actionable x 2 pii`) |
| states per label vector | median **52** (min 29, max 69) |
| distinct (template, label) | **1,534** (8 templates x 192), median 6, min 1, max 17 |

So the bank is ~6 renderings of each (template, label) pair — and the min of 1
means some pairs are effectively unseen. The label space is only 192 vectors.

## The grid

`generator.systematic: true` enumerates **every (latent, template) pair once**:

- `train` = 192 x 8 in-dist templates = **1,536**
- `test_indist` = 1,536 (same pairs, different rendering)
- `test_heldout` = 192 x 2 held-out templates = **384**
- breakout unchanged (4,000 / 500)

`configs/bge_grid_v3.yaml` (bge-m3 + `film: true` for the v3 head) runs it.

## Results — smaller, faster, BETTER

| dataset | doc states | extract | train | indist | **held-out** | breakout | paddle |
|---|---|---|---|---|---|---|---|
| random bank (full) | 12,000 | ~60 min | ~70 min | 0.954 | 0.916 | — | 0.955 |
| random bank (reduced samples) | 12,000 | cache | 18 min | 0.942 | 0.765 | — | 0.956 |
| **grid** | **3,456** | **4.9 min** | **4.3 min** | 0.924 | **0.913** | **0.988** | **0.976** |

**3.5x fewer states, ~10x faster cycle (~11 min end to end), higher held-out and
breakout.** In-dist is slightly lower (0.924 vs 0.954) — fewer redundant samples
means less memorisation, which is the right trade.

Sanity note: the earlier 0.765 for the *reduced-sample random* run is **not** a
data-size effect — the grid uses the same reduced training-time settings and
reaches 0.913. The random bank's problem was **coverage** (per-pair counts range
1..17), not volume.

## `is_actionable`: the instability was coverage, not the contradictory cues

The random bank showed `is_actionable` swinging 0.50..0.98 across runs. Two
manual-audit defects were found in the generator:

- **email subject** hard-coded `"Action required: pending approval"` for
  `urgency=high` regardless of `actionable` → 155 train emails whose subject
  contradicts `gold=False`; in 44 of them (quality=0) the subject is the ONLY
  action cue → unlearnable. Fixed: the subject's action prefix now tracks
  `actionable`.
- **markdown** `"- [x| ] Follow-up required"` — the label hinged on a single
  checkbox character (650 cases). Fixed: the phrase itself now differs
  (`Follow-up required` vs `No follow-up required`).

**But the grid alone already fixed `is_actionable`** (0.974 in-dist / 0.990
held-out, balanced True 0.97 / False 0.98) — so the *dominant* cause was the
random bank's uneven coverage, not the contradictory cues. The cue fix (both
principled) measured **no additional gain** (0.989 / 0.952 post-fix; inside the
~3-8pp seed noise). Keep the fix; the effect is unproven.

## Operating notes

- Grid cycles are ~11 min (generate 1 + extract 5 + train 4 + eval 1) — fast
  enough for seed repeats and A/Bs.
- **Training can get stuck at the stage-1 plateau**: one grid run early-stopped
  at epoch 24 with val_acc 0.484 (chance). A retrain gave 0.973. Fixed by
  gating patience-based early stop on the escape guard (commit `1c98e16`); such
  runs now return `None` for the caller to retry.
- `data_grid/` and `results_grid*/` are gitignored.

## Next

- Re-run v3-vs-v4 and the sub-path (explicit-routing) experiment on the grid.
- v3-vs-v4 on **Qwen** at the same config, to confirm the "architecture gap is
  Qwen-only" claim without the schedule confound.
