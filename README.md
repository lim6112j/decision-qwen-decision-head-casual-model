# Decision Lab

**Benchmark: Qwen Decision Head vs Prompt-based Causal LM for gridworld decisions.**

Compares three agents on the same backbone (`Qwen3.5-0.8B`, GGUF via llama-server):

| Agent | Approach | Training |
|---|---|---|
| `head_trained` | Frozen backbone + trained MLP head | Yes (head only) |
| `head_random` | Frozen backbone + random MLP head | No |
| `prompt_lm_zero_shot` | Prompt-based causal LM | No (zero-shot) |

## Setup

```bash
# 1. Install llama.cpp (brew or nix)
brew install llama.cpp

# 2. Create venv + install Python deps
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Download model
bash scripts/setup.sh
```

## Usage

```bash
source .venv/bin/activate

# Step by step:
python -m decision_lab generate   # synthetic gridworld data
python -m decision_lab extract    # feature extraction (needs llama-server)
python -m decision_lab train      # train decision head
python -m decision_lab eval       # benchmark all agents
python -m decision_lab report     # generate report

# Or run everything:
python -m decision_lab all

# Or:
bash scripts/run_all.sh

# Interactive web simulator (select one of the 3 models, watch live episodes):
python -m decision_lab ui         # opens http://127.0.0.1:8000
```

## Configuration

Edit `configs/default.yaml` to adjust gridworld size, model path, head hyperparameters, etc.

## HTTP API

Other projects can call the trained dynamic head over HTTP (no code import):
start the UI service and `POST /api/decide-dynamic` with your own text and
question configs — see [docs/http-api.md](docs/http-api.md).

## Tests

```bash
pytest tests/ -v
```

## Results

Output to `results/`:
- `report.md` — human-readable comparison with tables
- `metrics.json` — raw numbers for further analysis

## Latest Benchmark Run (2026-10-09, text classification, v2 field-set head)

Full pipeline (extract → train → eval → report) on Apple Silicon (MPS), llama.cpp server, Qwen3.5-0.8B UD-Q4_K_XL GGUF. Datasets: 10,000 train / 1,000 in-dist test / 1,000 held-out test (text states with typed question labels).

**Training:** 8 in-dist templates (email, ticket, json, text, log_entry, chat_message, markdown, bullet_list)  
**Held-out:** 2 unseen templates (report, config_file) — test for format generalization

> **Held-out formats are not inherently hard.** They are hand-coded Python renderer
> functions in `states/generator.py` — just like the in-dist ones. The drop
> happens because the head has literally never seen embeddings from these formats
> during training. Moving `report` and `config_file` into in-dist and retraining
> would bring them up to ~90% accuracy like the others. The hold-out exists only
> to measure generalization to unseen text structures, not because those formats
> are fundamentally difficult.

### Current Benchmark

| Agent | In-dist acc | Held-out acc | Mean latency | Parse failures |
|---|---|---|---|---|
| `head_trained` | 93.3% | 48.8% | 0.2 ms | 0 |
| `head_dynamic` (v2) | **84.6%** | **73.9%** | 11.0 ms | 0 |
| `head_random` | 40.2% | 39.3% | 0.2 ms | 0 |
| `prompt_lm_zero_shot` | 60.5% | 64.5% | ~368 ms | 0 |

Previous run (v1 pooled-state head, 2026-10-08): head_dynamic 82.2% in-dist / **39.1% held-out**; typed head and prompt LM numbers unchanged by the restructure (their pipeline still uses the pooled caches). Comparison caveat: v2 changes both the head and the state input pipeline (field splitting), so the held-out gain (+34.8pp) reflects the structural change as a whole — which is exactly what it was designed to do (see "v2 head" below).

Key findings:

- **Typed head dominates in-dist** (93.3%) at ~1,800× lower latency than prompt LM — but drops 44pp on held-out formats (48.8%). The head overfits to surface text patterns in the training templates.
- **Prompt LM inverts**: does *better* on held-out (64.5% vs 60.5%) because it reads text directly rather than relying on format-dependent embeddings.
- **Dynamic head v2** (84.6% in-dist, 73.9% held-out) supports any number of options at inference without retraining — and field-set attention recovers most of the held-out gap the v1 pooled head suffered (39.1% → 73.9%), now beating the prompt LM held-out (64.5%) too.
- **Qwen backbone is never modified.** Training only touches the head MLP (~400K params for typed, ~430K for the v2 dynamic head). The GGUF model file (`models/Qwen3.5-0.8B-UD-Q4_K_XL.gguf`) is served read-only by llama-server for embedding extraction and chat completion. The backbone never sees gradients — all training happens on pre-extracted frozen embeddings.
- **LoRA adapter on frozen embeddings does not help** held-out generalization (v1 experiment). A post-hoc low-rank transformation (rank=64) trained on in-dist embeddings can only re-weight existing dimensions — it cannot bridge the fundamental gap between embeddings of different text formats. Held-out actually regressed (48.7% → 43.5% with adapter). The bottleneck is the frozen Qwen3.5-0.8B backbone's format-specific embedding geometry — the v2 field-level input attacks the same problem from the input side instead.

### Generalization: dynamic head on novel option sets (zero retraining)

The dynamic head was trained on labeled option sets (e.g. sentiment options: positive/negative/neutral). At inference, it accepts any option text:

| Novel option set | Options | Accuracy |
|---|---|---|
| Binary sentiment (Good/Bad) | 2 | **90.0%** |
| 5-option sentiment scale | 5 | **77.0%** |
| 7-option Likert agreement | 7 | **63.0%** |
| Star ratings (1star→5star) | 5 | 42.0% |
| Novel metaphorical labels | 4 | 7.0% |
| *TypedHead baseline (fixed bank)* | — | *95.7%* |

The head generalizes when option texts are semantically adjacent to training labels (sentiment synonyms), but fails on completely novel domains — attention relies on backbone embeddings of option texts being in a similar semantic space to training.

### Breakout paddle-direction (trained domain, 2026-10-09)

#### Problem

Calling the trained dynamic head over HTTP with a Breakout game state:

```json
{
  "custom_text": "The ball is clearly to the LEFT of the paddle (gap 124 px) and moving right and up, away from the paddle.",
  "questions": [{"type": "choice", "options": ["left", "right", "stay"],
                 "question": "which direction the paddle move?"}]
}
```

expected `left` (move toward the ball) but returned **`right` for almost every state**.

#### Root cause (not a decode bug)

1. **The head was never trained on Breakout.** Training data was synthetic
   office documents (sentiment/urgency/quality); the option strings
   `left`/`right`/`stay` never appeared during training.
2. **The `question` string is metadata only** — `POST /api/decide-dynamic`
   discards it. The decision reduces to: embed the state text and argmax it
   against bare embeddings of the option strings. With an out-of-distribution
   state text and untrained option keys, generic embedding geometry decides —
   and the frequent, polysemous token "right" wins nearly always.

The `options` *are* used (embedded via the backbone, scored against the state
via attention — only the `question` text is dropped). They were simply
meaningless before retraining.

#### Fix

- New generator `states/breakout.py`: Breakout state texts with gold paddle
  direction computed from geometry — move **toward the ball's horizontal
  position**, `stay` within ~24 px. Ball velocity words are sampled
  independently of the ball's side, so motion phrases ("moving right",
  "away from the paddle") carry no shortcut to the label.
- 4,000 states mixed into **dynamic-head training only** (`breakout_weight: 3`,
  ~24% of training samples). Typed-head training is untouched — its metrics
  are byte-identical. Document JSONL stays byte-identical for the same seed
  (breakout uses `Random(seed + 1)`), so doc feature caches remain valid.
- Synonym variants for `paddle_direction` added to `CHOICE_SYNONYM_MAPS`
  (shuffled option order + label synonyms → permutation invariance).
- Bias is now measurable: `AgentMetrics` carries per-label accuracy and a
  confusion matrix; a `test_breakout` benchmark pass was added.

#### Results

v2 field-set head (2026-10-09):

| Split | Mean acc | left | stay | right | ECE |
|---|---|---|---|---|---|
| `test_breakout` (head_dynamic v2) | **98.0%** | 100% | 95.7% | 98.5% | 0.011 |

(v1 pooled head was 95.5% mean / 95.5 / 95.7 / 95.4 — the field-level input
removed the last 2.5pp.) No single-label collapse; the exact state above
returns `left` in both option orders. One honest observation: that exact
state (gap 124 px, motion pointing the other way) comes out as a near-uniform
distribution with `left` winning argmax only marginally — the head signals
low confidence where motion words contradict geometry, rather than
confidently collapsing. Document-question accuracy is preserved (in-dist
84.6% vs 82.2% on v1; sentiment 100%, urgency 92.5%).

#### Gotchas found during this work

- **Last-token pooling makes text order matter.** The head reads decisive
  information from the *end* of the state text. An initial template set with
  status-first/geometry-last produced a head that failed on real
  geometry-first states. The prose template therefore randomizes sentence
  order and sometimes omits the status sentence entirely — coverage of the
  shapes real clients send beats one canonical layout.
- **The HTTP server caches the checkpoint at startup.** `python -m
  decision_lab ui` loads `models/head_dynamic.pt` once (`webapp/agents.py`
  `_load_dynamic_agent`); after retraining, restart the server or HTTP probes
  silently serve the stale head.
- **Training split hygiene**: the dynamic-head training loop previously
  trained on val/calibration samples during the generalize phase (inflating
  reported val_acc). Both curriculum phases now train on the train split only.

#### Deferred: feeding the `question` string to the model

The `question`-is-metadata-only caveat stands. Fixing it means
question-conditioned state embeddings on **both** sides: training embeds
states via the pre-extracted `features_*.npz` cache (would require
re-extracting features per (state × question) pair), while inference embeds
`custom_text` live. Payoff: one head could answer different questions over
the same state text. Until then, all question-specific meaning must live in
the state text — put the geometry in `custom_text`, not in `question`.

Note: option sets from domains outside office documents and Breakout (e.g.
ratings, metaphors) remain unlearned — see the novel-option table above.

Sanity check: `python scripts/quick_breakout_test.py`.

### v2 head: field-set cross-attention (2026-10-09)

The pooled-vector input was the structural ceiling: the state text was
collapsed into **one** embedding, so one out-of-distribution token polluted
the whole state representation and the head matched options against a single
point in embedding space (a learned decision boundary around a handful of
format islands — any format drift re-collapses the head).

v2 (checkpoint arch `state_set: true`) feeds the state as a **set of
field/sentence embeddings** and options cross-attend over that set:

```
fields (B, M, 1024) → shared field_enc → field keys (B, M, d_k)
options (N, 1024)   → opt_query        → queries (B, N, d_k)
α_ij = softmax over fields(q_j·k_i/√d_k), masked padding excluded
z_j = Σ α_ij·field_i → shared score_head → logits (B, N)
```

- Contamination is **localized**: an OOD token corrupts only the field it
  appears in. Structured formats split per leaf (`"ball.x: 369"`), log lines
  per `KEY=VALUE` token, prose per sentence — syntax tokens no longer dilute
  the geometry.
- The head can match *"the ball is to the right"* ↔ the `right` option at
  field granularity.
- The full-text embedding is always prepended as **field 0** (summary field,
  `dynamic_head.include_summary_field`) — global context + a guaranteed
  non-empty set.
- **Chunking consistency invariant**: training and inference must split the
  state identically. One source of truth — `states/fields.py` (`state_fields`
  / `split_state_fields`) — is used by feature extraction, the training
  pipeline, the serving agent, and the HTTP default for `custom_text`.
  Renderer-emitted fields (breakout jsonl `fields` key) are asserted equal to
  the heuristic split in `tests/test_fields.py`, so callers who omit
  `custom_fields` still get training-matched chunking.
- New ragged cache `data/features_{split}_fields.npz` (`features (ΣM_i, 1024)`
  + `field_counts`); the pooled `features_*.npz` caches remain untouched and
  still feed the typed head.
- v1 checkpoints load unchanged (arch flag fallback; v1 backup kept at
  `models/head_dynamic_v1.pt`). The dead untrained `noul_head` path is gone
  from the pipeline — noul is scored via the same attention path it trains on.

Results of the restructure: `test_breakout` 95.5% → **98.0%**, in-dist
82.2% → **84.6%**, held-out 39.1% → **73.9%** (+34.8pp — the largest gain is
exactly where the pooled-vector input was weakest), at 11 ms latency and
ECE ≤ 0.05 everywhere.

> Deeper notes on how the decision heads train (frozen-backbone design,
> last-token pooling pitfalls, data decorrelation, split hygiene):
> [docs/decision-head-training.md](docs/decision-head-training.md).

Quick sanity check without the full pipeline (one question per test set, all three agents):

```bash
python scripts/quick_one_state_test.py 1
```