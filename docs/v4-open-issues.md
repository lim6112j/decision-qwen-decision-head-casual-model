# v4 open issues

Deliberately deferred (2026-10-11). The v4 head is shipped as-is: it solves
the problem it was built for (unseen question targets) and trades ~4pp of
breakout `paddle_direction` against v3. Everything below is an improvement,
not a defect.

Scores in this file are from the 200-state benchmark subsample unless noted.

## 1. Close the remaining breakout `paddle_direction` gap (94.0% vs v3's 98.0%)

The adversarial slice — states where the ball's motion word disagrees with
its position side — is at 89.5%, vs 100% on the agreeing slice. Error mass
is concentrated in `gold=left → right` and `gold=stay → right`.

Ideas, roughly in order of expected payoff per unit of effort:

- **Breakout sample share.** The full run collapses ~14k states into 344k
  samples, of which ~240k are shape-augmented document states. Breakout is
  `4000 × breakout_weight 3 = 12k` states over 5 qids. Raising
  `breakout_weight` to 6-8 rebalances the ratio without touching the bank.
  Evidence this matters: the reduced run (1.5k doc / 1.5k breakout, *no*
  shape augmentation, 45 epochs) reached 0.830 with the shift-only design,
  while the full run with the same design reached 0.745 — more data made
  this task worse, which points at dilution rather than capacity.
- **A `stay`-specific probe.** "stay" needs a magnitude comparison
  (`|gap| ≤ 24 px`) that neither of the other answers needs. Recentring the
  option queries does not obviously help a threshold decision; an explicit
  numeric field (e.g. a learned scale on the `GAP=` token) may.
- **Question-gate capacity.** `gate`/`shift` are single `Linear(hidden → d_k)`
  layers. A two-layer MLP (gate only, keeping shift linear) is cheap to try.

## 2. Held-out document formats (75.7%, v3 71.5%)

`quality` (56.0%) and `contains_pii` (76.0%) carry the loss — the held-out
templates render the same latents in shapes the field splitter chunks
differently. Not v4-specific; inherited from v2.

## 3. Serving-time OOD guard (not implemented)

The plan called for snapping an unseen question embedding to the nearest
trained one (or falling back to the zero question) when similarity is low.
Dropped from the v4 branch by decision — the v4 head degrades gracefully
instead (a random untrained question embedding gives finite, sane-shaped
logits; see `test_unseen_question_type_smoke`), so the guard is a
nice-to-have, not a correctness fix. Note the UI already flags near-uniform
distributions as unreliable (`web/static/app.js`).

## 4. Embedding cache for training runs

`generate_dynamic_training_data` re-embeds ~24k texts (option, question and
shape-variant field texts) on every run — ~20 minutes per retrain, the
dominant cost of an iteration. The per-state field features are already
cached in `data/features_*_fields.npz`; these runtime embeddings are not.
Persisting them (keyed by text + gguf path) would make iteration ~4× faster.
Caveat: vectors depend on the served backend (local Metal vs a tunneled
remote differed at cosine 0.9997 — small but real), so the cache key must
include the backend, not just the text.

## 5. Operational notes worth keeping

- **`http_proxy` traps localhost.** With `http_proxy` set and no `no_proxy`
  entry, every embedding request to `127.0.0.1` rides the proxy: measured
  6.4 texts/s vs 37.3 with `no_proxy=localhost,127.0.0.1`. Run training and
  evaluation with `no_proxy` set.
- **`LlamaServer` adopts whatever answers `/health` on its port** (added so
  a tunneled remote can be reused). Consequence: a config pointing at a port
  where a *different* backend is listening silently trains on that backend's
  embeddings. Keep extraction and training on the same port; the v4 training
  config uses 8090 for this reason.
- **Retrain then restart the UI.** The webapp loads the checkpoint once at
  startup (`webapp/agents.py`); a retrain does not hot-reload.
