"""Benchmark harness: typed-question accuracy, confidence calibration (ECE),
and latency for each agent on each test set."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from decision_lab.config import Config
from decision_lab.head.model import (
    TypedDecisionHead,
    create_random_head,
    get_device,
    is_correct,
    load_head,
    predict_all,
)
from decision_lab.states.dataset import load_dataset

# Latency is measured on a subsample; accuracy is measured on every state.
ECE_BINS = 10
HEAD_WARMUP_RUNS = 3
HEAD_TIMED_RUNS = 5


@dataclass
class EvalRow:
    agent_id: str
    doc_id: int
    predictions: dict            # {qid: predicted value (missing qid = parse failure)}
    gold_labels: dict            # {qid: gold value}
    confidence: dict             # {qid: calibrated confidence float}
    latency_ms: float            # decision wall clock ms
    raw_output: str = ""         # LM output text (empty for head agents)


@dataclass
class AgentMetrics:
    mean_accuracy: float                    # across all questions
    per_question_accuracy: dict             # {qid: float}
    mean_confidence: float
    ece: float                              # expected calibration error (all questions pooled)
    ece_per_question: dict                  # {qid: float}
    mean_latency_ms: float
    median_latency_ms: float
    p99_latency_ms: float
    num_parse_failures: int = 0             # (row, question) pairs with no answer
    avg_tokens: float = 0.0
    raw_outputs: list[str] = field(default_factory=list)


def benchmark_head(
    states,
    features: np.ndarray,
    head: TypedDecisionHead,
    temperatures: dict,
    agent_id: str,
    device: torch.device,
) -> list[EvalRow]:
    """Run a typed-head agent (trained or random) on all states."""
    head.eval()
    head.to(device)
    results = []

    for i, s in enumerate(states):
        feat = torch.tensor(features[i], dtype=torch.float32, device=device).unsqueeze(0)

        # Warmup + timed runs (single forward pass answers all questions)
        latencies = []
        with torch.no_grad():
            for _ in range(HEAD_WARMUP_RUNS):
                _ = head(feat)
            outputs = None
            for _ in range(HEAD_TIMED_RUNS):
                t0 = time.perf_counter()
                outputs = head(feat)
                latencies.append((time.perf_counter() - t0) * 1000)

        decoded = predict_all(head.question_spec, outputs, temperatures)
        results.append(EvalRow(
            agent_id=agent_id,
            doc_id=s.doc_id,
            predictions={qid: d["predicted"] for qid, d in decoded.items()},
            gold_labels=dict(s.labels),
            confidence={qid: d["confidence"] for qid, d in decoded.items()},
            latency_ms=float(np.mean(latencies)),
        ))

    return results


def benchmark_prompt(
    states,
    agent,
    agent_id: str,
    warmup: int,
    timed_runs: int,
    latency_sample_size: int,
) -> list[EvalRow]:
    """Run prompt-based agent on all states.

    Accuracy from a single decision per state; latency from warmup+timed runs
    on the first `latency_sample_size` states (LLM calls dominate wall time).
    """
    results = []

    for i, s in enumerate(states):
        if i < latency_sample_size:
            for _ in range(warmup):
                agent.decide(s)
            latencies = []
            answers, raw = {}, ""
            for _ in range(timed_runs):
                t0 = time.perf_counter()
                answers, raw = agent.decide(s)
                latencies.append((time.perf_counter() - t0) * 1000)
            latency = float(np.mean(latencies))
        else:
            answers, raw = agent.decide(s)
            latency = 0.0

        results.append(EvalRow(
            agent_id=agent_id,
            doc_id=s.doc_id,
            predictions=answers,
            gold_labels=dict(s.labels),
            confidence={},
            latency_ms=latency,
            raw_output=raw,
        ))

    return results


def expected_calibration_error(confidences: np.ndarray, corrects: np.ndarray, bins: int = ECE_BINS) -> float:
    """ECE: Σ_bins |acc(bin) − conf(bin)| × (bin_size / total)."""
    if len(confidences) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_acc = corrects[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += abs(bin_acc - bin_conf) * (mask.sum() / len(confidences))
    return float(ece)


def compute_metrics(rows: list[EvalRow], question_spec: dict) -> AgentMetrics:
    """Aggregate eval rows into summary metrics (per-question accuracy + ECE)."""
    total = len(rows)
    if total == 0:
        return AgentMetrics(
            mean_accuracy=0.0, per_question_accuracy={}, mean_confidence=0.0,
            ece=0.0, ece_per_question={}, mean_latency_ms=0.0,
            median_latency_ms=0.0, p99_latency_ms=0.0,
        )

    per_q_acc: dict[str, float] = {}
    per_q_ece: dict[str, float] = {}
    all_confs, all_corrects = [], []
    parse_failures = 0

    for qid, spec_entry in question_spec.items():
        corrects = []
        confs = []
        for r in rows:
            ok = is_correct(spec_entry, r.predictions.get(qid), r.gold_labels.get(qid))
            corrects.append(ok)
            if qid not in r.predictions:
                parse_failures += 1
            conf = r.confidence.get(qid)
            if conf is not None:
                confs.append((conf, ok))
        corrects_arr = np.array(corrects, dtype=float)
        per_q_acc[qid] = float(corrects_arr.mean())
        if confs:
            conf_arr = np.array([c for c, _ in confs])
            ok_arr = np.array([ok for _, ok in confs], dtype=float)
            per_q_ece[qid] = expected_calibration_error(conf_arr, ok_arr)
            all_confs.extend(conf_arr.tolist())
            all_corrects.extend(ok_arr.tolist())
        else:
            per_q_ece[qid] = 0.0

    if all_confs:
        ece = expected_calibration_error(np.array(all_confs), np.array(all_corrects, dtype=float))
        mean_conf = float(np.mean(all_confs))
    else:
        ece, mean_conf = 0.0, 0.0

    latencies = np.array([r.latency_ms for r in rows if r.latency_ms > 0])
    if len(latencies) == 0:
        latencies = np.array([0.0])
    tokens = [len(r.raw_output.split()) if r.raw_output else 0 for r in rows]

    return AgentMetrics(
        mean_accuracy=float(np.mean(list(per_q_acc.values()))),
        per_question_accuracy=per_q_acc,
        mean_confidence=mean_conf,
        ece=ece,
        ece_per_question=per_q_ece,
        mean_latency_ms=float(latencies.mean()),
        median_latency_ms=float(np.median(latencies)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        num_parse_failures=parse_failures,
        avg_tokens=float(np.mean(tokens)),
        raw_outputs=[r.raw_output for r in rows if r.raw_output][:5],
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
    from decision_lab.head.model import build_question_spec
    from decision_lab.prompt_lm.agent import PromptAgent, PromptMode

    device = get_device()
    bc = cfg.benchmark
    question_spec = build_question_spec(cfg.questions)
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
        if bc.max_test_states > 0 and len(states) > bc.max_test_states:
            rng = np.random.RandomState(cfg.generator.seed)
            idx = rng.choice(len(states), size=bc.max_test_states, replace=False)
            idx.sort()
            print(f"  [{test_name}] subsampling {len(idx)}/{len(states)} states")
            states = [states[i] for i in idx]
            if features is not None:
                features = features[idx]   # feature rows follow dataset file order

        all_metrics[test_name] = {}

        # 1. Head trained (temperatures come from the checkpoint)
        head_trained = load_head(str(models_dir / "head_trained.pt"), device=device)
        print(f"  [{test_name}] benchmarking head_trained...")
        rows = benchmark_head(states, features, head_trained, head_trained.temperatures,
                              "head_trained", device)
        all_metrics[test_name]["head_trained"] = compute_metrics(rows, question_spec)

        # 2. Head random
        head_random = create_random_head(question_spec, hidden_dim=cfg.head.hidden_dim,
                                         dropout=cfg.head.dropout)
        print(f"  [{test_name}] benchmarking head_random...")
        rows = benchmark_head(states, features, head_random, {}, "head_random", device)
        all_metrics[test_name]["head_random"] = compute_metrics(rows, question_spec)

        # 3. Prompt LM zero-shot
        agent_zero = PromptAgent(cfg, server, mode=PromptMode.ZERO_SHOT)
        print(f"  [{test_name}] benchmarking prompt_lm_zero_shot...")
        rows = benchmark_prompt(states, agent_zero, "prompt_lm_zero_shot",
                                bc.warmup_runs, bc.timed_runs, bc.latency_sample_size)
        all_metrics[test_name]["prompt_lm_zero_shot"] = compute_metrics(rows, question_spec)

    return all_metrics