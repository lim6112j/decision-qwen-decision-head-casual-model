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

## Latest Benchmark Run (2026-10-07)

Full pipeline (`python -m decision_lab all`) on Apple Silicon (MPS), llama.cpp server, Qwen3.5-0.8B UD-Q4_K_XL GGUF. Wall time: ~51 min. Datasets: 1,200 train / 300 in-dist test / 300 held-out test states (held-out layouts are 8×8–10×10 with higher wall density).

| Agent | In-dist acc | Held-out acc | Mean latency | Parse failures |
|---|---|---|---|---|
| `head_trained` | **99.3%** | **89.0%** | 0.4 ms | 0 |
| `head_random` | 0.0% | 5.7% | 0.3 ms | 0 |
| `prompt_lm_zero_shot` | 5.7% | 5.3% | ~90 ms | 32 / 16 of 300 |

Key findings:

- **Trained head dominates the zero-shot LM**: 99.3% vs 5.7% accuracy at ~200× lower latency.
- **Generalization gap is the headline number**: head accuracy drops 10.3pp on unseen larger layouts (99.3% → 89.0%).
- **Zero-shot LM is at chance with a "Right" bias**: nearly all outputs are `Answer: Right` regardless of the grid state (avg 2.1 tokens).
- **Qwen3.5 is a thinking model**: without `enable_thinking: false` (passed as `chat_template_kwargs` in `LlamaServer.chat`), all 128 `max_tokens` are consumed inside `reasoning_content` and `content` comes back empty — every LM decision parse-fails. The flag is required for this benchmark.
- Head training: best val_acc 0.996, early stop at epoch 57.

Quick sanity check without the full pipeline (one question per test set, all three agents):

```bash
python scripts/quick_one_state_test.py 1
```