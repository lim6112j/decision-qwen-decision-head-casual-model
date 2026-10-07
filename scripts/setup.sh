#!/usr/bin/env bash
# setup.sh — Download model GGUF + verify llama.cpp availability.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$PROJECT_DIR/models"
QUANT="UD-Q4_K_XL"
MODEL_REPO="unsloth/Qwen3.5-0.8B-GGUF"

echo "=== Decision Lab Setup ==="

# 1. Check llama-server
if command -v llama-server &>/dev/null; then
    echo "✓ llama-server found: $(which llama-server)"
else
    echo "⚠ llama-server not found in PATH."
    echo "  Install llama.cpp with your nix config, or: brew install llama.cpp"
fi

# 2. Check huggingface CLI (huggingface_hub >= 1.0 ships `hf`; older versions ship `huggingface-cli`)
if command -v hf &>/dev/null; then
    HF_CLI="hf"
elif command -v huggingface-cli &>/dev/null; then
    HF_CLI="huggingface-cli"
else
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
    HF_CLI="hf"
fi

# 3. Download model
GGUF_FILE="$MODELS_DIR/Qwen3.5-0.8B-${QUANT}.gguf"
if [ -f "$GGUF_FILE" ]; then
    echo "✓ Model already downloaded: $GGUF_FILE"
else
    mkdir -p "$MODELS_DIR"
    echo "Downloading $MODEL_REPO ($QUANT)..."
    "$HF_CLI" download "$MODEL_REPO" \
        --include "*${QUANT}*" \
        --local-dir "$MODELS_DIR"
    echo "✓ Downloaded to $MODELS_DIR"
fi

echo "=== Setup complete ==="
echo "Run: scripts/run_all.sh"