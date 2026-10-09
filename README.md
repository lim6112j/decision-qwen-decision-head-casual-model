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

## Latest Benchmark Run (2026-10-08, text classification)

Full pipeline (extract → train → eval → report) on Apple Silicon (MPS), llama.cpp server, Qwen3.5-0.8B UD-Q4_K_XL GGUF. Datasets: 10,000 train / 1,000 in-dist test / 1,000 held-out test (text states with typed question labels).

**Training:** 8 in-dist templates (email, ticket, json, text, log_entry, chat_message, markdown, bullet_list)  
**Held-out:** 2 unseen templates (report, config_file) — test for format generalization

> **Held-out formats are not inherently hard.** They are hand-coded Python renderer
> functions in `states/generator.py` — just like the in-dist ones. The 44pp drop
> happens because the head has literally never seen embeddings from these formats
> during training. Moving `report` and `config_file` into in-dist and retraining
> would bring them up to ~90% accuracy like the others. The hold-out exists only
> to measure generalization to unseen text structures, not because those formats
> are fundamentally difficult.

### Current Benchmark

| Agent | In-dist acc | Held-out acc | Mean latency | Parse failures |
|---|---|---|---|---|
| `head_trained` | 92.5% | 48.7% | 0.2 ms | 0 |
| `head_dynamic` | 82.2% | 39.1% | 8.5 ms | 0 |
| `head_random` | 41.6% | 39.2% | 0.2 ms | 0 |
| `prompt_lm_zero_shot` | 60.5% | 64.5% | ~360 ms | 0 |

Key findings:

- **Typed head dominates in-dist** (92.5%) at ~1,800× lower latency than prompt LM — but drops 44pp on held-out formats (48.7%). The head overfits to surface text patterns in the training templates.
- **Prompt LM inverts**: does *better* on held-out (64.5% vs 60.5%) because it reads text directly rather than relying on format-dependent embeddings.
- **Dynamic head** (82.2% in-dist, 39.1% held-out) supports any number of options at inference without retraining — trade ~10pp in-dist for full flexibility.
- **Qwen backbone is never modified.** Training only touches the head MLP (~400K params for typed, ~100K for dynamic). The GGUF model file (`models/Qwen3.5-0.8B-UD-Q4_K_XL.gguf`) is served read-only by llama-server for embedding extraction and chat completion. The backbone never sees gradients — all training happens on pre-extracted frozen embeddings.
- **LoRA adapter on frozen embeddings does not help** held-out generalization. A post-hoc low-rank transformation (rank=64) trained on in-dist embeddings can only re-weight existing dimensions — it cannot bridge the fundamental gap between embeddings of different text formats. Held-out actually regressed (48.7% → 43.5% with adapter). The bottleneck is the frozen Qwen3.5-0.8B backbone, which produces format-specific embeddings that no amount of post-hoc linear transformation can align.

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

| Split | Mean acc | left | stay | right |
|---|---|---|---|---|
| `test_breakout` (head_dynamic) | 95.5% | 95.5% | 95.7% | 95.4% |

The confusion matrix shows no single-label collapse, and the exact state
above now returns `left` at 0.999 confidence (stable under shuffled option
order). Original document-question accuracy is preserved (in-dist 81.8% vs
82.0% before mixing; sentiment 100%, urgency 92.5%).

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

> Deeper notes on how the decision heads train (frozen-backbone design,
> last-token pooling pitfalls, data decorrelation, split hygiene):
> [docs/decision-head-training.md](docs/decision-head-training.md).

Quick sanity check without the full pipeline (one question per test set, all three agents):

```bash
python scripts/quick_one_state_test.py 1
```