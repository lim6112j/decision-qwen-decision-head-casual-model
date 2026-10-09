"""Generate comparison report: markdown + JSON from typed-question metrics."""

import json
from dataclasses import asdict
from pathlib import Path

from decision_lab.eval.benchmark import AgentMetrics

AGENT_ORDER = ("head_trained", "head_random", "prompt_lm_zero_shot")
AGENT_LABELS = {
    "head_trained": "Typed Head (trained)",
    "head_random": "Typed Head (random)",
    "prompt_lm_zero_shot": "Prompt LM (zero-shot)",
}


def generate_report(
    all_metrics: dict[str, dict[str, AgentMetrics]],
    results_dir: Path,
) -> None:
    """Write results/report.md and results/metrics.json."""
    results_dir.mkdir(parents=True, exist_ok=True)

    json_out = {
        test: {agent: asdict(m) for agent, m in agents.items()}
        for test, agents in all_metrics.items()
    }
    (results_dir / "metrics.json").write_text(json.dumps(json_out, indent=2, ensure_ascii=False))

    lines = _build_markdown(all_metrics)
    (results_dir / "report.md").write_text("\n".join(lines))


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def _build_markdown(all_metrics: dict[str, dict[str, AgentMetrics]]) -> list[str]:
    lines = [
        "# Typed Decision Head vs Causal LM Benchmark Results",
        "",
        "Questions per state: choice (option pick), score (rubric level),",
        "noul (boolean P(true)) — all answered in a single forward pass by the head.",
        "",
    ]

    for test_name, agents in all_metrics.items():
        lines += [f"## {test_name}", "", "### Overall", ""]
        lines.append(
            "| Agent | Mean Accuracy | Mean Confidence | ECE | Mean Latency (ms) | "
            "Median (ms) | P99 (ms) | Parse Failures | Avg Tokens |"
        )
        lines.append(
            "|-------|--------------|-----------------|-----|-------------------|"
            "-------------|----------|----------------|------------|"
        )
        for agent_id in AGENT_ORDER:
            m = agents.get(agent_id)
            if m is None:
                continue
            lines.append(
                f"| {AGENT_LABELS[agent_id]} | {_fmt_pct(m.mean_accuracy)} | "
                f"{m.mean_confidence:.3f} | {m.ece:.3f} | {m.mean_latency_ms:.1f} | "
                f"{m.median_latency_ms:.1f} | {m.p99_latency_ms:.1f} | "
                f"{m.num_parse_failures} | {m.avg_tokens:.1f} |"
            )

        lines += ["", "### Per-question accuracy", ""]
        qids = _all_question_ids(agents)
        if qids:
            lines.append("| Agent | " + " | ".join(qids) + " |")
            lines.append("|-------" + "|-------" * len(qids) + "|")
            for agent_id in AGENT_ORDER:
                m = agents.get(agent_id)
                if m is None:
                    continue
                cells = " | ".join(_fmt_pct(m.per_question_accuracy.get(q, 0.0)) for q in qids)
                lines.append(f"| {AGENT_LABELS[agent_id]} | {cells} |")

        lines += ["", "### Per-question ECE (lower = better calibrated)", ""]
        if qids:
            lines.append("| Agent | " + " | ".join(qids) + " |")
            lines.append("|-------" + "|-------" * len(qids) + "|")
            for agent_id in AGENT_ORDER:
                m = agents.get(agent_id)
                if m is None:
                    continue
                cells = " | ".join(f"{m.ece_per_question.get(q, 0.0):.3f}" for q in qids)
                lines.append(f"| {AGENT_LABELS[agent_id]} | {cells} |")
        lines.append("")

    lines += _comparison_dimensions(all_metrics)
    lines += _generalization(all_metrics)
    lines += _breakout_section(all_metrics)
    lines += _sample_outputs(all_metrics)
    return lines


def _all_question_ids(agents: dict[str, AgentMetrics]) -> list[str]:
    for agent_id in AGENT_ORDER:
        m = agents.get(agent_id)
        if m and m.per_question_accuracy:
            return list(m.per_question_accuracy.keys())
    return []


def _comparison_dimensions(all_metrics) -> list[str]:
    ref = all_metrics.get("test_indist", {})
    if not ref:
        return []
    ht = ref.get("head_trained")
    hr = ref.get("head_random")
    pz = ref.get("prompt_lm_zero_shot")

    def fmt(fn, a) -> str:
        return fn(a) if a else "—"

    def fmt_acc(a):
        return f"{a.mean_accuracy:.1%}"

    def fmt_lat(a):
        return f"{a.median_latency_ms:.0f}ms"

    def fmt_ece(a):
        return f"{a.ece:.3f}"

    def fmt_conf(a):
        return f"{a.mean_confidence:.3f}"

    def fmt_fails(a):
        return f"{a.num_parse_failures}"

    def fmt_toks(a):
        return f"{a.avg_tokens:.0f}"

    return [
        "## Comparison Dimensions (test_indist)",
        "",
        "| Dimension | Typed Head (trained) | Typed Head (random) | Prompt LM (zero-shot) |",
        "|-----------|---------------------|---------------------|----------------------|",
        f"| Inference Speed | {fmt(fmt_lat, ht)} | {fmt(fmt_lat, hr)} | {fmt(fmt_lat, pz)} |",
        f"| Accuracy | {fmt(fmt_acc, ht)} | {fmt(fmt_acc, hr)} | {fmt(fmt_acc, pz)} |",
        f"| Calibration (ECE) | {fmt(fmt_ece, ht)} | {fmt(fmt_ece, hr)} | {fmt(fmt_ece, pz)} |",
        f"| Mean Confidence | {fmt(fmt_conf, ht)} | {fmt(fmt_conf, hr)} | {fmt(fmt_conf, pz)} |",
        "| Typed Outputs | Yes (probs + confidence) | Yes (probs + confidence) | No (text only) |",
        "| Pre-training Required | Yes (head MLP) | No | No (zero-shot) |",
        f"| Parse Failures | 0 | 0 | {fmt(fmt_fails, pz)} |",
        f"| Avg Tokens Generated | 0 | 0 | {fmt(fmt_toks, pz)} |",
        "",
    ]


def _generalization(all_metrics) -> list[str]:
    if "test_heldout" not in all_metrics or "test_indist" not in all_metrics:
        return []
    h = all_metrics["test_heldout"]
    i = all_metrics["test_indist"]

    lines = [
        "## Generalization (unseen state templates)",
        "",
        "| Agent | Accuracy (indist) | Accuracy (heldout) | Drop |",
        "|-------|-------------------|--------------------|------|",
    ]
    for agent_id in AGENT_ORDER:
        mi, mh = i.get(agent_id), h.get(agent_id)
        if mi is None or mh is None:
            continue
        drop = (mi.mean_accuracy - mh.mean_accuracy) * 100
        lines.append(
            f"| {AGENT_LABELS[agent_id]} | {_fmt_pct(mi.mean_accuracy)} | "
            f"{_fmt_pct(mh.mean_accuracy)} | {drop:+.1f}pp |"
        )
    lines.append("")
    return lines


def _breakout_section(all_metrics) -> list[str]:
    """Per-label accuracy + confusion matrix for the Breakout paddle question.

    Exposes label collapse: a head that answers "right" to everything shows
    per-label accuracy ~0 for left/stay and ~1 for right.
    """
    m = all_metrics.get("test_breakout", {}).get("head_dynamic")
    if m is None:
        return []

    lines = [
        "## Breakout paddle-direction (test_breakout, head_dynamic)",
        "",
        f"Mean accuracy: {_fmt_pct(m.mean_accuracy)}",
        "",
        "### Per-label accuracy (gold label →)",
        "",
        "| Gold label | Accuracy |",
        "|------------|----------|",
    ]
    for label, acc in sorted(m.per_label_accuracy.get("paddle_direction", {}).items()):
        lines.append(f"| {label} | {_fmt_pct(acc)} |")

    conf = m.confusion.get("paddle_direction", {})
    if conf:
        pred_labels = sorted({p for row in conf.values() for p in row})
        lines += [
            "",
            "### Confusion matrix (rows = gold, columns = predicted)",
            "",
            "| gold \\ pred | " + " | ".join(pred_labels) + " |",
            "|---" + "|---" * len(pred_labels) + "|",
        ]
        for gold in sorted(conf):
            cells = " | ".join(str(conf[gold].get(p, 0)) for p in pred_labels)
            lines.append(f"| {gold} | {cells} |")
    lines.append("")
    return lines


def _sample_outputs(all_metrics) -> list[str]:
    pz = all_metrics.get("test_indist", {}).get("prompt_lm_zero_shot")
    if not pz or not pz.raw_outputs:
        return []
    lines = ["## Sample Prompt LM Outputs", ""]
    for i, out in enumerate(pz.raw_outputs[:5]):
        lines += [f"### Example {i + 1}", "```", out, "```", ""]
    return lines