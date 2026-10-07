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

Quick sanity check without the full pipeline (one question per test set, all three agents):

```bash
python scripts/quick_one_state_test.py 1
```