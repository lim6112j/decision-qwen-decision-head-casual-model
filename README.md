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

## Running a specific dynamic-head version (v3 / v4)

The dynamic head has two shipped architectures. They are **not
interchangeable** — v4's code refuses `arch_version <= 3` on load by design
(clean break: the FiLM modules v3 needs were deleted), so each version needs
its own checkout.

| version | code | checkpoint | question conditioning |
|---|---|---|---|
| v3 | tag `v3-head` (`a1abd9a`) | `models/head_dynamic.pt` | FiLM on the option queries |
| v4 | `main` after the v4 merge | `models/head_dynamic_v4.pt` | question-field attention (answers unseen question targets) |

`configs/default.yaml` → `dynamic_head.checkpoint_filename` selects which
checkpoint the webapp and the benchmark load. Both checkpoints can live side
by side in `models/`; **`models/` is gitignored**, so a fresh checkout does
not bring the `.pt` files with it.

Running v3 after v4 lands — no retraining needed, the checkpoint is a file:

```bash
git worktree add ../decision-v3 v3-head          # v3-era code
cp models/head_dynamic.pt ../decision-v3/models/ # models/ is gitignored
cd ../decision-v3 && python -m decision_lab ui   # separate port / venv
```

Verified 2026-10-11: the code at tag `v3-head` loads
`models/head_dynamic.pt` and runs a forward pass unchanged.

Two more things worth knowing:

- **One dynamic head per process.** `build_agents` registers a single
  `head_dynamic`; switching versions means switching the config (or running
  two checkouts on different ports), not selecting per request.
- **Evaluation needs the feature caches** (`data/features_*_fields.npz`).
  They are gitignored too; a fresh checkout can regenerate them with
  `python -m decision_lab extract`, which costs llama-server embedding time
  (see `docs/v4-open-issues.md` for the `http_proxy` trap that makes this
  much slower than it should be).

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
| `head_dynamic` (v4) | **94.3%** | **75.7%** | ~11 ms | 0 |
| `head_random` | 40.2% | 39.3% | 0.2 ms | 0 |
| `prompt_lm_zero_shot` | 60.5% | 64.5% | ~368 ms | 0 |

`head_dynamic` re-measured on the v4 checkpoint (2026-10-11, same 200-state
subsample, fixed 6-question bank); the other rows are unchanged by the v4
work (typed head and prompt LM share none of the changed code paths) and
carry their v2-run numbers. Latency is not re-measured — the v4 forward adds
one attention row and two small linears per option (well under a millisecond)
to a path dominated by field embedding.

Previous run (v1 pooled-state head, 2026-10-08): head_dynamic 82.2% in-dist / **39.1% held-out**; typed head and prompt LM numbers unchanged by the restructure (their pipeline still uses the pooled caches). Comparison caveat: v2 changes both the head and the state input pipeline (field splitting), so the held-out gain (+34.8pp) reflects the structural change as a whole — which is exactly what it was designed to do (see "v2 head" below).

Key findings:

- **Typed head dominates in-dist** (93.3%) at ~1,800× lower latency than prompt LM — but drops 44pp on held-out formats (48.8%). The head overfits to surface text patterns in the training templates.
- **Prompt LM inverts**: does *better* on held-out (64.5% vs 60.5%) because it reads text directly rather than relying on format-dependent embeddings.
- **Dynamic head v4** (94.3% in-dist, 75.7% held-out) supports any number of options at inference without retraining — and field-set attention recovers most of the held-out gap the v1 pooled head suffered (39.1% → 75.7%), now beating the prompt LM held-out (64.5%) too. v4 additionally answers question targets it was never trained on (see "v4 head" below).
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
2. **The `question` string was metadata only (pre-v3)** — `POST /api/decide-dynamic`
   discarded it. The decision reduced to: embed the state text and argmax it
   against bare embeddings of the option strings. With an out-of-distribution
   state text and untrained option keys, generic embedding geometry decided —
   and the frequent, polysemous token "right" won nearly always. (Since v3 the
   question string conditions the head — see "v3 head: question-conditioned
   queries" below.)

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

#### Deferred: feeding the `question` string to the model → done in v3

This was implemented as the **v3 head** (question-conditioned queries —
see "v3 head: question-conditioned queries" below). The approach chosen is
query-side FiLM fusion on pre-extracted embeddings, not per-(state ×
question) feature re-extraction, so no new feature caches were needed.

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

### v3 head: question-conditioned queries (2026-10-09)

In v2 the `question` string was still metadata: options were embedded as
bare texts and the same state+options pair always produced the same answer
regardless of what was being asked. v3 (checkpoint `arch_version: 3`) makes
the question functional via **query-side FiLM fusion** — the question text
is embedded (same frozen backbone as state/option texts) and modulates each
option's query before it reads the field set:

```
Q_opt = normalize(opt_query(opt_emb))                    # v2 unchanged
gate  = q_mod_scale(question_emb)  ─ zero-init Linear(1024 → d_k)
shift = q_mod_bias(question_emb)   ─ zero-init Linear(1024 → d_k)
Q     = normalize((1 + gate) · Q_opt + shift)
```

- **Identity init**: both linears start at zero, so v3's forward is exactly
  v2's for any question — the cosine-attention O(1) logit spread (the v2
  anti-stall design) is preserved. The `(1 + gate)` form is load-bearing:
  a bare `gate · Q_opt` would be `normalize(0)` → NaN at init.
- **Shortcut-proof data**: breakout now has a second question,
  `ball_motion` (gold = sign of ball_vx, `stay` when vx = 0), sharing the
  *identical* option texts `left/right/stay` with `paddle_direction` while
  vx is decorrelated from side — the same state has different golds under
  the two questions, so the head cannot fit both without reading the
  question embedding. The document bank's three noul questions
  (`is_actionable`/`contains_pii`/`is_urgent`, same `false`/`true` options,
  independent golds) provide the same forcing for the doc domain.
  ~5% of training samples carry no question (learned null-question behavior
  for callers omitting `question`).
- **Checkpoint compat**: v2 checkpoints load with the fusion keys at
  identity init (warning printed) — question text is ignored until
  retrain; `python -m decision_lab ui` must be restarted after retraining
  as usual. v1 handling unchanged. The HTTP schema is unchanged
  (`question` was already optional).
- **Feature caches are fingerprinted**: `extract_features` /
  `extract_field_features` now store a SHA-256 fingerprint of the source
  dataset (`+ include_summary_field`) inside the .npz and re-extract
  automatically on mismatch ("cache stale (source changed)"). Before this,
  a regenerated dataset with an existing cache silently trained on
  embeddings of the OLD texts paired with the NEW labels — a nearly
  invisible corruption that masked the v3 fix for a full debugging day.
- Breaking the old "metadata only" rule: the same `custom_text` can now be
  asked different questions (e.g. paddle direction vs ball motion) and get
  different, correct answers.

Results of the v3 retrain: `test_breakout` **paddle_direction 98.0%** (v2
parity) **and ball_motion 99.5%** (new capability — same state, same option
texts, different gold), in-dist 82.2% → **95.7%**, held-out 73.9% → **71.5%**
(v2 parity within run variance). The v2-era tie on the contradictory-evidence
state (geometry says LEFT, motion says "moving right") is gone in the v3
head — verified live (2026-10-10): that state answers paddle_direction
`left` @ 0.9999 and ball_motion `right` @ 0.996, each question reading its
own cue instead of averaging the two.

> Deeper notes on how the decision heads train (frozen-backbone design,
> last-token pooling pitfalls, data decorrelation, split hygiene):
> [docs/decision-head-training.md](docs/decision-head-training.md).

Quick sanity check without the full pipeline (one question per test set, all three agents):

```bash
python scripts/quick_one_state_test.py 1
```
### v4 head: question-field attention (2026-10-11)

v3's question conditioning was a dense map of the question embedding
(`gate, shift = Linear(question_emb)`) trained on ~25 fixed question texts
(5 qids × 3-4 handwritten phrasings). A `Linear(1024 → d_k)` is only
regulated on the points it was trained on; in every direction orthogonal to
those 25 it is free. Any question target outside the bank therefore received
a near-random modulation. Measured: v3 answers the polarity *inversions* of
its own questions at chance or worse — `not_urgent` 3.0%, `no_pii` 4.0%,
`not_negative` 48.5% (it answers the base question).

v4 gives the question an actual computation path:

```
field_repr = field_enc(state)                     # (B, M, H)
K          = normalize(field_key(field_repr))     # (B, M, d_k)

Qq    = normalize(q_question(question_emb))       # question's own query
alpha = softmax(K · Qq · attn_scale_q)            # question reads the fields
z_q   = alpha @ field_repr                        # (B, H) attended context

Q_j   = normalize((1 + gate(z_q)) · opt_query(opt_j) + shift(z_q))
z_j   = attention(Q_j, K) @ field_repr
z_j'  = z_j + qz_readout([z_j ; z_q])             # zero-init residual
logit = score_head(z_j')
```

- **Why attention fixes the generalization.** The question's query is
  matched against field keys by cosine similarity, so an unseen question
  lands near a trained one in backbone space and reads the same fields —
  semantic similarity flows through a real computation path instead of
  being memorized. `gate`/`shift` (from `z_q`) are zero-init: at init v4 is
  exactly the v2 forward for any question, preserving the cosine-attention
  O(1) logit spread, and `(1 + gate)` avoids the `normalize(0)` NaN.
- **Why the gate is load-bearing.** An additive shift alone
  (`Q_j + shift(z_q)`) only *translates* every option query; it cannot
  re-point matching. A per-field attention bias shared across options fails
  the other way: small enough to be safe, the distractor field wins; large
  enough to steer, every option reads the same field and the logits
  collapse to a tie. The multiplicative gate reshapes each option's query
  per dimension — that is what lets the same option text ("left") match the
  ball's *position* field under `paddle_direction` and the *motion* field
  under `ball_motion`.

**Compositional question bank** (`head/question_bank.py`): 17 qids across
two domains — base predicates, polarity inversions, derived predicates, and
new question types over existing latents — each with ~40-50 phrasings
(hand-written cores × prefix modifiers; prefix-only, because the backbone
pools the last token). Golds come from `state.labels` or pure derivations,
never hand-labeled. Training draws ONE sample per (state, qid) pair, so the
sample count stays ~344k instead of the ~20M full cross-product.

**Results** (2026-10-11, v4 checkpoint, 200-state subsample):

| Split / question | v3 | v4 |
|---|---|---|
| `test_indist` (fixed 6-qid bank) | 95.7% | **94.3%** |
| `test_heldout` (fixed 6-qid bank) | 71.5% | **75.7%** |
| `test_breakout` `paddle_direction` | 98.0% | **94.0%** |
| `test_breakout` `ball_motion` | 99.5% | 99.5% |
| held-out phrasings, doc bank (`test_phrasings`) | — | **96.4%** |
| held-out phrasings, breakout (`test_phrasings`) | — | **97.3%** |

New-question targets, which v3 cannot answer at all:

| qid | v3 | v4 |
|---|---|---|
| `not_urgent` | 3.0% | **97.4%** |
| `no_pii` | 4.0% | **99.8%** |
| `not_actionable` | 47.5% | **96.5%** |
| `not_negative` | 48.5% | **99.9%** |
| `pii_presence` (new choice type) | 61.0% | **100%** |
| `contact_mentioned` (same gold, new text) | 70.5% | **100%** |

The adversarial-slice diagnostic (breakout states where the ball's motion
word disagrees with its position side — the "moving right, ball on the
LEFT" case) went from 48.9% (first v4 attempt) to **89.5%** across the
three question-conditioning designs.

Trade-off: v4 gives up ~4pp of v3's breakout `paddle_direction` (v3
overfit that domain with 2 questions and 3-4 phrasings). Remaining ideas
are tracked in `docs/v4-open-issues.md`.
