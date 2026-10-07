#!/usr/bin/env bash
# run_all.sh — Full pipeline: generate data → extract features → train → eval → report
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="python3"

echo "=== Decision Lab: Full Pipeline ==="
echo ""

# Generate synthetic data
$PYTHON -m decision_lab generate
echo ""

# Extract features via llama-server
$PYTHON -m decision_lab extract
echo ""

# Train head
$PYTHON -m decision_lab train
echo ""

# Evaluate all agents
$PYTHON -m decision_lab eval
echo ""

echo "=== Done ==="
echo "Results: results/report.md"
echo "Metrics: results/metrics.json"