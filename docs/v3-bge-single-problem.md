# v3 (FiLM) on bge-m3 — the single-problem optimum? (2026-10-14)

Recorded 2026-10-14 on branch `v3-bge` (off `main`). Question: for a system
that solves **one fixed problem** (breakout paddle control), which dynamic-head
architecture + backbone is best?

## Hypothesis being tested

v4 (question-field attention) exists to generalize to **unseen question
targets**. A single-problem system never uses that, and v4 pays for it: on
Qwen, breakout paddle measured **v3 98.0 vs v4 94.0** (a ~4pp "generalization
tax"). So the prediction was **v3 (FiLM) + bge ≥ 98.0**.

## What was built

- `DynamicDecisionHead(film=True)` — the v3 FiLM path (question embedding
  modulates the option queries `q_mod_scale/bias` and the read-out
  `z_mod_scale/bias`, all zero-init → starts as v2) restored as an **opt-in
  architecture** alongside v4 in the same codebase. `arch_version` 3 (film) /
  4 (attention); `dynamic_head.film` selects it. Commit `18e5290`.
- `configs/bge_v3.yaml` (film on) and `configs/bge_v4shrink.yaml` (film off) —
  **identical** otherwise, so the architecture can be A/B'd with the config held
  fixed. Both use the reduced schedule (`variants_per_question 1`,
  `breakout_weight 1`, `shape_augmentation false`) for ~18 min runs.
- bge-m3 backbone via the cherry-picked infra (`d820bf0`): CLS pooling, port
  8091, backend-aware cache fingerprint. The `data_bge/` feature caches were
  reused (fingerprint matched → no 60-min re-extraction).

## Results

**Architecture is a no-op on bge — v3 and v4 are identical to three digits:**

| config (architecture is the only difference) | paddle_direction | ball_motion | in-dist | held-out |
|---|---|---|---|---|
| **v3** (film) + bge, shrink | **0.956** | 1.000 | 0.943 | 0.765 |
| **v4** + bge, shrink | **0.956** | 1.000 | 0.942 | 0.765 |

**All breakout-paddle measurements to date:**

| config | paddle |
|---|---|
| **v3 + Qwen** (full schedule) | **0.980** |
| v3 + bge (shrink) | 0.956 |
| v4 + bge (shrink) | 0.956 |
| v4 + bge (full) | 0.955 |
| v4 + Qwen (full) | 0.940 |

## Conclusion

1. **The v3-vs-v4 gap is Qwen-specific.** On Qwen, FiLM beat question-field
   attention by ~4pp on breakout; on **bge the two are identical**.
2. **bge does not help breakout.** Its +15.9pp held-out gain was concentrated in
   the *document* questions (`quality` 56%→93%); breakout paddle actually reads
   slightly *worse* on bge (0.956) than v3+Qwen (0.980).
3. **The best measured breakout config is v3 (FiLM) + Qwen (0.980)** — not
   v3+bge. Backbone choice is **task-dependent**: bge for semantic document
   questions, Qwen's last-token pooling for the geometry/status-at-end breakout
   tasks.

So the single-problem recommendation is **v3 (FiLM) + Qwen**, and the
generalization-tax hypothesis holds **only on Qwen** — on bge it vanishes.

## Caveats

- v3+Qwen (0.980) and v4+Qwen (0.940) were measured in the v4 era with the
  **full** schedule; the bge rows use the reduced schedule. The reduced
  schedule did **not** cost breakout on bge (v4 full 0.955 ≈ v4 shrink 0.956),
  so the schedule is unlikely to explain the gap — but a clean rerun (v3/v4 on
  **Qwen**, same reduced config) has not been done.
- Seed variance on this head is ~3pp (documented in
  [typed-decisions-results.md](typed-decisions-results.md)); the v3=v4 equality
  is well inside it, the bge-vs-Qwen breakout gap (2.4pp) is borderline.
- The eval hit one HTTP 500 from a single over-long text (llama.cpp's default
  512-token physical batch); metrics still completed.

## Next

- Rerun v3 vs v4 on **Qwen** at the reduced config to confirm the "Qwen-only"
  architecture gap without the schedule confound.
- If breakout-only is the product, start from **v3 + Qwen**
  (`models/head_dynamic.pt` from the v3 era), not bge.
