"""Benchmark harness: measure accuracy + latency for each agent on each test set."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from decision_lab.config import Config
from decision_lab.env.dataset import load_dataset
from decision_lab.env.gridworld import GridState
from decision_lab.head.model import DecisionHead, get_device, load_head


@dataclass
class EvalRow:
    agent_id: str
    action_id: int | None   # predicted action, None if parse failed
    label: int               # ground truth action
    latency_ms: float        # decision wall clock ms
    raw_output: str = ""     # LM output text (empty for head agents)


@dataclass
class AgentMetrics:
    accuracy: float
    mean_latency_ms: float
    median_latency_ms: float
    p99_latency_ms: float
    num_parse_failures: int = 0
    avg_tokens: float = 0.0
    raw_outputs: list[str] = field(default_factory=list)


def benchmark_head(
    states: list[GridState],
    features: np.ndarray,
    head: DecisionHead,
    agent_id: str,
    warmup: int,
    timed_runs: int,
    device: torch.device,
) -> list[EvalRow]:
    """Run head-based agent (trained or random) on all states."""
    head.eval()
    head.to(device)
    results = []

    for i, s in enumerate(states):
        feat = torch.tensor(features[i], dtype=torch.float32, device=device).unsqueeze(0)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = head(feat).argmax(dim=1)

        # Timed runs
        latencies = []
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = head(feat).argmax(dim=1).item()
            latencies.append((time.perf_counter() - t0) * 1000)

        results.append(EvalRow(
            agent_id=agent_id,
            action_id=pred,
            label=s.label,
            latency_ms=np.mean(latencies),
        ))

    return results


def benchmark_prompt(
    states: list[GridState],
    agent,
    agent_id: str,
    warmup: int,
    timed_runs: int,
) -> list[EvalRow]:
    """Run prompt-based agent on all states."""
    results = []

    for s in states:
        # Warmup
        for _ in range(warmup):
            agent.decide(s)

        # Timed runs
        latencies = []
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            action, raw = agent.decide(s)
            latencies.append((time.perf_counter() - t0) * 1000)

        results.append(EvalRow(
            agent_id=agent_id,
            action_id=action,
            label=s.label,
            latency_ms=np.mean(latencies),
            raw_output=raw,
        ))

    return results


def compute_metrics(rows: list[EvalRow]) -> AgentMetrics:
    """Aggregate eval rows into summary metrics."""
    correct = sum(1 for r in rows if r.action_id == r.label)
    total = len(rows)
    parse_failures = sum(1 for r in rows if r.action_id is None)
    latencies = [r.latency_ms for r in rows]
    arr = np.array(latencies)
    tokens = [len(r.raw_output.split()) if r.raw_output else 0 for r in rows]

    if total == 0:
        return AgentMetrics(
            accuracy=0.0, mean_latency_ms=0.0, median_latency_ms=0.0,
            p99_latency_ms=0.0, num_parse_failures=0, avg_tokens=0.0,
        )

    return AgentMetrics(
        accuracy=correct / total,
        mean_latency_ms=float(arr.mean()),
        median_latency_ms=float(np.median(arr)),
        p99_latency_ms=float(np.percentile(arr, 99)),
        num_parse_failures=parse_failures,
        avg_tokens=float(np.mean(tokens)),
        raw_outputs=[r.raw_output for r in rows if r.raw_output],
    )


def run_benchmark(
    data_dir: Path,
    cfg: Config,
    server,
    models_dir: Path,
) -> dict[str, dict[str, AgentMetrics]]:
    """Run full benchmark: 3 agents × 2 test sets → metrics dict.

    Returns:
        {test_name: {agent_id: AgentMetrics}}
    """
    device = get_device()
    bc = cfg.benchmark
    test_splits = {
        "test_indist": data_dir / "test_indist.jsonl",
        "test_heldout": data_dir / "test_heldout.jsonl",
    }

    all_metrics: dict[str, dict[str, AgentMetrics]] = {}

    for test_name, test_path in test_splits.items():
        if not test_path.exists():
            print(f"  skip {test_name}: not found")
            continue

        states = load_dataset(test_path)
        feature_path = data_dir / f"features_{test_name}.npz"
        features = np.load(feature_path)["features"] if feature_path.exists() else None

        all_metrics[test_name] = {}

        # 1. Head trained
        head_trained = load_head(
            str(models_dir / "head_trained.pt"),
            hidden_dim=cfg.head.hidden_dim,
            num_actions=cfg.head.num_actions,
            dropout=cfg.head.dropout,
        )
        print(f"  [{test_name}] benchmarking head_trained...")
        rows = benchmark_head(states, features, head_trained, "head_trained", bc.warmup_runs, bc.timed_runs, device)
        all_metrics[test_name]["head_trained"] = compute_metrics(rows)

        # 2. Head random
        head_random = DecisionHead(
            hidden_dim=cfg.head.hidden_dim,
            num_actions=cfg.head.num_actions,
            dropout=cfg.head.dropout,
        )
        print(f"  [{test_name}] benchmarking head_random...")
        rows = benchmark_head(states, features, head_random, "head_random", bc.warmup_runs, bc.timed_runs, device)
        all_metrics[test_name]["head_random"] = compute_metrics(rows)

        # 3. Prompt LM zero-shot
        from decision_lab.prompt_lm.agent import PromptAgent, PromptMode
        agent_zero = PromptAgent(cfg, server, mode=PromptMode.ZERO_SHOT)
        print(f"  [{test_name}] benchmarking prompt_lm_zero_shot...")
        rows = benchmark_prompt(states, agent_zero, "prompt_lm_zero_shot", bc.warmup_runs, bc.timed_runs)
        all_metrics[test_name]["prompt_lm_zero_shot"] = compute_metrics(rows)

    return all_metrics