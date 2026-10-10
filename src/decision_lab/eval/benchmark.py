"""Benchmark harness: typed-question accuracy, confidence calibration (ECE),
and latency for each agent on each test set."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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
from decision_lab.head.dynamic_model import (
    DynamicDecisionHead,
    create_random_dynamic_head,
    decode_dynamic_answer,
    is_correct_dynamic,
    load_dynamic_head,
    make_choice_question,
    make_noul_question,
    make_score_question,
    question_option_texts,
)
from decision_lab.head.dynamic_train import default_question_text
from decision_lab.backbone.features import load_field_features
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
    # Per-gold-label accuracy {qid: {label: acc}} — exposes label collapse
    # (e.g. always predicting "right" → per_label_accuracy["left"] ≈ 0).
    per_label_accuracy: dict = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)   # {qid: {gold: {pred: count}}}


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


def benchmark_dynamic_head(
    states,
    field_sets: Sequence[np.ndarray],
    head: DynamicDecisionHead,
    question_spec: dict,
    option_emb_cache: dict[str, torch.Tensor],
    temperature: float,
    agent_id: str,
    device: torch.device,
) -> list[EvalRow]:
    """Run the dynamic head agent on all states using the fixed question bank.

    Args:
        field_sets: per-state (M_i, input_dim) field-set embeddings (v2) —
            a v1 head reads field 0, which is the full-text embedding.
        option_emb_cache: {option_text: (input_dim,) tensor on device} —
            pre-embedded option texts from the backbone.
    """
    head.eval()
    head.to(device)
    results = []

    for i, s in enumerate(states):
        feat = torch.from_numpy(np.asarray(field_sets[i], dtype=np.float32)).unsqueeze(0).to(device)

        # Warmup + timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(HEAD_WARMUP_RUNS):
                _run_dynamic_forward(head, feat, question_spec, option_emb_cache)
            for _ in range(HEAD_TIMED_RUNS):
                t0 = time.perf_counter()
                decoded = _run_dynamic_forward(
                    head, feat, question_spec, option_emb_cache, temperature,
                )
                latencies.append((time.perf_counter() - t0) * 1000)

        results.append(EvalRow(
            agent_id=agent_id,
            doc_id=s.doc_id,
            predictions={qid: d["predicted"] for qid, d in decoded.items()},
            gold_labels=dict(s.labels),
            confidence={qid: d["confidence"] for qid, d in decoded.items()},
            latency_ms=float(np.mean(latencies)),
        ))

    return results


def _run_dynamic_forward(
    head: DynamicDecisionHead,
    state_tensor: torch.Tensor,
    question_spec: dict,
    option_emb_cache: dict[str, torch.Tensor],
    temperature: float = 1.0,
) -> dict:
    """Run dynamic head on one state against the fixed question bank.

    v4: each question's text provides the field-selection query — the cache
    must contain the question strings as well as option texts (see
    _collect_option_texts).

    Returns {qid: decoded answer dict} — same format as predict_all for TypedDecisionHead.
    """
    results = {}
    for qid, spec in question_spec.items():
        kind = spec["type"]
        if kind == "noul":
            question_text = spec.get("question", qid)
            opt_embs = torch.stack([
                option_emb_cache["false"], option_emb_cache["true"]
            ])  # (2, D)
            scores = head.forward_choice(
                state_tensor, opt_embs, question_emb=option_emb_cache[question_text],
            )
            results[qid] = decode_dynamic_answer(
                make_noul_question(question_text), scores, temperature,
            )
        elif kind == "choice":
            option_texts = spec["options"]
            question_text = default_question_text(qid)
            opt_embs = torch.stack(
                [option_emb_cache[t] for t in option_texts]
            )  # (n_opts, D)
            scores = head.forward_choice(
                state_tensor, opt_embs, question_emb=option_emb_cache[question_text],
            )
            results[qid] = decode_dynamic_answer(
                make_choice_question(option_texts, question_text), scores, temperature,
            )
        else:  # score
            level_texts = spec["levels"]
            question_text = default_question_text(qid)
            opt_embs = torch.stack(
                [option_emb_cache[t] for t in level_texts]
            )  # (n_levels, D)
            scores = head.forward_score(
                state_tensor, opt_embs, question_emb=option_emb_cache[question_text],
            )
            results[qid] = decode_dynamic_answer(
                make_score_question(level_texts, question_text), scores, temperature,
            )
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
    per_label_acc: dict[str, dict] = {}
    confusion: dict[str, dict] = {}
    all_confs, all_corrects = [], []
    parse_failures = 0

    for qid, spec_entry in question_spec.items():
        corrects = []
        confs = []
        label_hits: dict[str, list] = {}
        gold_pred_counts: dict[str, dict] = {}
        for r in rows:
            pred = r.predictions.get(qid)
            gold = r.gold_labels.get(qid)
            ok = is_correct(spec_entry, pred, gold)
            corrects.append(ok)
            if qid not in r.predictions:
                parse_failures += 1
            conf = r.confidence.get(qid)
            if conf is not None:
                confs.append((conf, ok))
            if gold is not None:
                gk, pk = str(gold), str(pred)
                gold_pred_counts.setdefault(gk, {})
                gold_pred_counts[gk][pk] = gold_pred_counts[gk].get(pk, 0) + 1
                label_hits.setdefault(gk, []).append(ok)
        corrects_arr = np.array(corrects, dtype=float)
        per_q_acc[qid] = float(corrects_arr.mean())
        per_label_acc[qid] = {
            g: float(np.mean(hits)) for g, hits in label_hits.items()
        }
        confusion[qid] = gold_pred_counts
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
        per_label_accuracy=per_label_acc,
        confusion=confusion,
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
        field_path = data_dir / f"features_{test_name}_fields.npz"
        field_sets = load_field_features(field_path)[0] if field_path.exists() else None
        if bc.max_test_states > 0 and len(states) > bc.max_test_states:
            rng = np.random.RandomState(cfg.generator.seed)
            idx = rng.choice(len(states), size=bc.max_test_states, replace=False)
            idx.sort()
            print(f"  [{test_name}] subsampling {len(idx)}/{len(states)} states")
            states = [states[i] for i in idx]
            if features is not None:
                features = features[idx]   # feature rows follow dataset file order
            if field_sets is not None:
                field_sets = [field_sets[i] for i in idx]

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

        # 3. Dynamic head (attention-based, option values as inputs)
        # Pre-embed all option texts via backbone
        unique_option_texts = _collect_option_texts(question_spec)
        option_emb_cache_np = _embed_options_batch(server, unique_option_texts)
        option_emb_cache = {
            t: torch.tensor(emb, dtype=torch.float32, device=device)
            for t, emb in option_emb_cache_np.items()
        }

        dynamic_head, dynamic_temp = _load_dynamic_for_benchmark(cfg, models_dir, device)
        if field_sets is None and dynamic_head.state_set:
            raise RuntimeError(
                f"missing {field_path} for the v2 dynamic head — run `python -m decision_lab extract`"
            )
        dynamic_features = field_sets if dynamic_head.state_set else features
        print(f"  [{test_name}] benchmarking head_dynamic...")
        rows = benchmark_dynamic_head(
            states, dynamic_features, dynamic_head, question_spec,
            option_emb_cache, dynamic_temp, "head_dynamic", device,
        )
        all_metrics[test_name]["head_dynamic"] = compute_metrics(rows, question_spec)

        # 4. Prompt LM zero-shot
        agent_zero = PromptAgent(cfg, server, mode=PromptMode.ZERO_SHOT)
        print(f"  [{test_name}] benchmarking prompt_lm_zero_shot...")
        rows = benchmark_prompt(states, agent_zero, "prompt_lm_zero_shot",
                                bc.warmup_runs, bc.timed_runs, bc.latency_sample_size)
        all_metrics[test_name]["prompt_lm_zero_shot"] = compute_metrics(rows, question_spec)

    breakout_metrics = _benchmark_dynamic_breakout(data_dir, cfg, server, models_dir, bc)
    if breakout_metrics is not None:
        all_metrics["test_breakout"] = breakout_metrics

    # v4 held-out phrasing eval: the fixed-bank rows above only test the
    # canonical phrasing; the compositional bank's non-canonical phrasings
    # (and its new qids) measure unseen-question-target generalization.
    phrasing_metrics = _benchmark_dynamic_phrasings(
        data_dir, cfg, server, models_dir, bc, question_spec,
    )
    if phrasing_metrics is not None:
        all_metrics["test_phrasings"] = phrasing_metrics

    return all_metrics


def _benchmark_dynamic_phrasings(
    data_dir: Path,
    cfg: Config,
    server,
    models_dir: Path,
    bc,
    question_spec: dict,
) -> dict[str, AgentMetrics] | None:
    """Benchmark the v4 head on NON-canonical bank phrasings + new qids.

    For each test split (indist, breakout): every bank entry applicable to
    the split's states is evaluated under its held-out phrasings (entries
    1..n) plus — for qids outside the fixed config bank (inversions, new
    types, breakout derived predicates) — also its canonical phrasing,
    which no other benchmark row covers. Golds come from the bank's
    programmatic gold functions (exact by construction).
    """
    from decision_lab.head.question_bank import build_question_bank

    dynamic_head, dynamic_temp = _load_dynamic_for_benchmark(cfg, models_dir, device=get_device())
    if not dynamic_head.state_set:
        return None

    device = get_device()
    bank = build_question_bank()
    all_metrics: dict[str, AgentMetrics] = {}

    splits = [
        ("test_indist", data_dir / "test_indist.jsonl",
         data_dir / "features_test_indist_fields.npz"),
        ("test_breakout", data_dir / "test_breakout.jsonl",
         data_dir / "features_test_breakout_fields.npz"),
    ]

    for split_name, test_path, field_path in splits:
        if not test_path.exists() or not field_path.exists():
            continue
        states = load_dataset(test_path)
        field_sets = load_field_features(field_path)[0]
        if bc.max_test_states > 0 and len(states) > bc.max_test_states:
            rng = np.random.RandomState(cfg.generator.seed)
            idx = rng.choice(len(states), size=bc.max_test_states, replace=False)
            idx.sort()
            states = [states[i] for i in idx]
            field_sets = [field_sets[i] for i in idx]

        # All phrasings + option texts the bank will need on this split
        spec_by_qid = {}
        for entry in bank:
            if any(entry.applicable(s) for s in states):
                for phrasing in entry.phrasings:
                    spec_by_qid.setdefault(entry.qid, entry.config(phrasing))
        option_emb_cache = {
            t: torch.tensor(emb, dtype=torch.float32, device=device)
            for t, emb in _embed_options_batch(
                server, sorted(_collect_all_bank_texts(bank, spec_by_qid)),
            ).items()
        }

        # Evaluate phrasing i >= 1 for every qid; qids absent from the fixed
        # config bank are also evaluated at phrasing 0.
        fixed_qids = set(question_spec)
        per_q_correct: dict[str, int] = {}
        per_q_total: dict[str, int] = {}
        for i, state in enumerate(states):
            state_tensor = torch.tensor(field_sets[i], dtype=torch.float32, device=device)
            state_tensor = state_tensor.unsqueeze(0)  # (1, M, D)
            for entry in bank:
                if not entry.applicable(state):
                    continue
                for phrasing_idx, phrasing in enumerate(entry.phrasings):
                    if phrasing_idx == 0 and entry.qid in fixed_qids:
                        continue   # canonical already covered by fixed-bank rows
                    q = entry.config(phrasing)
                    texts = question_option_texts(q)
                    opt_embs = torch.stack([option_emb_cache[t] for t in texts])
                    q_emb = option_emb_cache[phrasing]
                    with torch.no_grad():
                        scores = dynamic_head.forward_choice(
                            state_tensor, opt_embs, question_emb=q_emb,
                        )
                    decoded = decode_dynamic_answer(q, scores, dynamic_temp)
                    gold = entry.gold(state)
                    correct = is_correct_dynamic(q, decoded["predicted"], gold)
                    per_q_correct.setdefault(entry.qid, 0)
                    per_q_total.setdefault(entry.qid, 0)
                    per_q_correct[entry.qid] += int(correct)
                    per_q_total[entry.qid] += 1

        if not per_q_total:
            continue
        per_q_acc = {
            qid: per_q_correct[qid] / per_q_total[qid] for qid in per_q_total
        }
        all_metrics[f"{split_name}"] = AgentMetrics(
            mean_accuracy=float(np.mean(list(per_q_acc.values()))),
            per_question_accuracy=per_q_acc,
            mean_confidence=0.0, ece=0.0, ece_per_question={},
            mean_latency_ms=0.0, median_latency_ms=0.0, p99_latency_ms=0.0,
        )
    return all_metrics or None


def _collect_all_bank_texts(bank, spec_by_qid: dict) -> list[str]:
    """Every option/level text AND phrasing the paraphrase eval will embed."""
    texts: set[str] = {"false", "true"}
    for entry in bank:
        if entry.qid in spec_by_qid:
            texts.update(entry.options or ())
            texts.update(entry.phrasings)
    return texts


def _benchmark_dynamic_breakout(
    data_dir: Path,
    cfg: Config,
    server,
    models_dir: Path,
    bc,
) -> dict[str, AgentMetrics] | None:
    """Benchmark the dynamic head on Breakout paddle-direction states.

    Only head_dynamic runs here: the typed head and PromptAgent have no
    paddle_direction question in their fixed banks.
    """
    from decision_lab.states.breakout import breakout_question_spec

    test_path = data_dir / "test_breakout.jsonl"
    feature_path = data_dir / "features_test_breakout.npz"
    field_path = data_dir / "features_test_breakout_fields.npz"
    if not test_path.exists() or not feature_path.exists():
        return None

    states = load_dataset(test_path)
    features = np.load(feature_path)["features"]
    field_sets = load_field_features(field_path)[0] if field_path.exists() else None
    if bc.max_test_states > 0 and len(states) > bc.max_test_states:
        rng = np.random.RandomState(cfg.generator.seed)
        idx = rng.choice(len(states), size=bc.max_test_states, replace=False)
        idx.sort()
        print(f"  [test_breakout] subsampling {len(idx)}/{len(states)} states")
        states = [states[i] for i in idx]
        features = features[idx]
        if field_sets is not None:
            field_sets = [field_sets[i] for i in idx]

    device = get_device()
    breakout_spec = breakout_question_spec()
    option_emb_cache = {
        t: torch.tensor(emb, dtype=torch.float32, device=device)
        for t, emb in _embed_options_batch(server, _collect_option_texts(breakout_spec)).items()
    }

    dynamic_head, dynamic_temp = _load_dynamic_for_benchmark(cfg, models_dir, device)
    if field_sets is None and dynamic_head.state_set:
        raise RuntimeError(
            f"missing {field_path} for the v2 dynamic head — run `python -m decision_lab extract`"
        )
    dynamic_features = field_sets if dynamic_head.state_set else features
    print("  [test_breakout] benchmarking head_dynamic...")
    rows = benchmark_dynamic_head(
        states, dynamic_features, dynamic_head, breakout_spec,
        option_emb_cache, dynamic_temp, "head_dynamic", device,
    )
    return {"head_dynamic": compute_metrics(rows, breakout_spec)}


def _collect_option_texts(question_spec: dict) -> list[str]:
    """Collect all unique option/level texts AND question strings.

    v3: question texts are embedded like options and looked up per question
    at forward time.
    """
    texts: set[str] = set()
    for qid, spec in question_spec.items():
        kind = spec["type"]
        if kind == "choice":
            texts.update(spec["options"])
            texts.add(default_question_text(qid))
        elif kind == "score":
            texts.update(spec["levels"])
            texts.add(default_question_text(qid))
        elif kind == "noul":
            texts.update(["false", "true"])
            if spec.get("question"):
                texts.add(spec["question"])
    return sorted(texts)


def _embed_options_batch(server, option_texts: list[str]) -> dict[str, np.ndarray]:
    """Batch-embed option texts via the backbone."""
    result = {}
    batch_size = 64
    for i in range(0, len(option_texts), batch_size):
        batch = option_texts[i : i + batch_size]
        embs = server.embed(batch)
        for text, emb in zip(batch, embs):
            result[text] = np.array(emb, dtype=np.float32)
    return result


def _load_dynamic_for_benchmark(
    cfg: Config, models_dir: Path, device: torch.device,
) -> tuple[DynamicDecisionHead, float]:
    """Load trained dynamic head checkpoint or fall back to random init.

    A missing checkpoint falls back to random init (the honest "untrained"
    baseline). A checkpoint that exists but fails to load is an *error*
    (stale/corrupt artifact) and is raised rather than silently degrading
    to random weights.
    """
    path = models_dir / cfg.dynamic_head.checkpoint_filename
    try:
        head = load_dynamic_head(str(path), device=device)
    except FileNotFoundError:
        print(f"  no dynamic head checkpoint at {path}; using random init")
        head = create_random_dynamic_head(
            hidden_dim=cfg.dynamic_head.hidden_dim,
            d_k=cfg.dynamic_head.d_k,
            dropout=cfg.dynamic_head.dropout,
            state_set=True,
        ).to(device)
        temperature = 1.0
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"dynamic head checkpoint at {path} exists but failed to load: {exc}. "
            f"This indicates a stale or corrupt checkpoint — retrain with "
            f"`python -m decision_lab train-dynamic` rather than silently "
            f"using random init."
        ) from exc
    else:
        temperature = getattr(head, "temperature", 1.0)
        print(f"  loaded dynamic head from {path} (T={temperature:.3f})")
    return head, temperature