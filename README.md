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

## Latest Benchmark Run (2026-10-07, wall-restored)

> **Correction (2026-10-07):** an earlier version of this table reported 99.3% / 89.0% for `head_trained`.
> Those numbers were an artifact of a bug in `load_dataset` (`env/dataset.py`): layouts were rebuilt with the
> goal but **without walls**, so feature extraction, training, and eval all ran on wall-free grids — a
> trivially learnable geometry task. The bug is fixed; the pipeline was fully re-run
> (re-extract → retrain → re-eval). Numbers below are the corrected ones.

Full pipeline (extract → train → eval → report) on Apple Silicon (MPS), llama.cpp server, Qwen3.5-0.8B UD-Q4_K_XL GGUF. Datasets: 1,200 train / 300 in-dist test / 300 held-out test states (held-out layouts are 8×8–10×10 with higher wall density).

| Agent | In-dist acc | Held-out acc | Mean latency | Parse failures |
|---|---|---|---|---|
| `head_trained` | **64.3%** | **46.0%** | 0.4 ms | 0 |
| `head_random` | 16.3% | 23.7% | 0.6 ms | 0 |
| `prompt_lm_zero_shot` | 8.7% | 2.3% | ~96 ms | 64 / 93 of 300 |

Key findings:

- **Trained head still clearly outperforms the zero-shot LM** (64.3% vs 8.7% in-dist) at ~250× lower latency — but the margin is far smaller than the wall-free numbers suggested.
- **Navigating around walls is genuinely hard for the head**: with walls actually present in the state renders, a 2-layer MLP on frozen backbone embeddings reaches only 0.629 val accuracy (best, early stop at epoch 16). Chance is 20%.
- **Generalization gap**: head accuracy drops 18.3pp on unseen larger layouts (64.3% → 46.0%).
- **Zero-shot LM is at chance with a "Right" bias**: nearly all outputs are `Answer: Right` regardless of the grid state (avg 2.2 tokens), and roughly 1 in 5 outputs fails to parse as an action.
- **Qwen3.5 is a thinking model**: without `enable_thinking: false` (passed as `chat_template_kwargs` in `LlamaServer.chat`), all 128 `max_tokens` are consumed inside `reasoning_content` and `content` comes back empty — every LM decision parse-fails. The flag is required for this benchmark.

Quick sanity check without the full pipeline (one question per test set, all three agents):

```bash
python scripts/quick_one_state_test.py 1
```