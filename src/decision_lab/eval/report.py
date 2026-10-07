"""Generate comparison report: markdown + JSON from benchmark metrics."""

import json
from dataclasses import asdict
from pathlib import Path

from decision_lab.eval.benchmark import AgentMetrics


def generate_report(
    all_metrics: dict[str, dict[str, AgentMetrics]],
    results_dir: Path,
) -> None:
    """Write results/report.md and results/metrics.json."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_out = {
        test: {agent: asdict(m) for agent, m in agents.items()}
        for test, agents in all_metrics.items()
    }
    (results_dir / "metrics.json").write_text(json.dumps(json_out, indent=2, ensure_ascii=False))

    # Markdown
    lines = _build_markdown(all_metrics)
    (results_dir / "report.md").write_text("\n".join(lines))


def _build_markdown(all_metrics: dict[str, dict[str, AgentMetrics]]) -> list[str]:
    lines = [
        "# Decision Head vs Causal LM Benchmark Results",
        "",
        "## Summary Table",
        "",
    ]

    # Build comparison table per test set
    for test_name, agents in all_metrics.items():
        lines.append(f"### {test_name}")
        lines.append("")
        lines.append(
            "| Agent | Accuracy | Mean Latency (ms) | Median Latency (ms) | "
            "P99 Latency (ms) | Parse Failures | Avg Tokens |"
        )
        lines.append(
            "|-------|----------|-------------------|---------------------|"
            "-----------------|----------------|------------|"
        )

        for agent_id in ["head_trained", "head_random", "prompt_lm_zero_shot"]:
            m = agents.get(agent_id)
            if m is None:
                continue
            lines.append(
                f"| {agent_id} | {m.accuracy:.4f} | {m.mean_latency_ms:.1f} | "
                f"{m.median_latency_ms:.1f} | {m.p99_latency_ms:.1f} | "
                f"{m.num_parse_failures} | {m.avg_tokens:.1f} |"
            )

        lines.append("")

    # Comparison dimensions (matching user's table)
    lines.append("## Comparison Dimensions")
    lines.append("")

    # Find reference metrics (in-dist test set)
    ref_test = "test_indist"
    if ref_test in all_metrics:
        m = all_metrics[ref_test]
        ht = m.get("head_trained")
        hr = m.get("head_random")
        pz = m.get("prompt_lm_zero_shot")

        lines.append("| Dimension | Head (trained) | Head (random) | Prompt LM (zero-shot) |")
        lines.append("|-----------|---------------|---------------|----------------------|")

        def fmt_acc(a: AgentMetrics | None) -> str:
            return f"{a.accuracy:.1%}" if a else "—"

        def fmt_lat(a: AgentMetrics | None) -> str:
            return f"{a.median_latency_ms:.0f}ms" if a else "—"

        def fmt_toks(a: AgentMetrics | None) -> str:
            return f"{a.avg_tokens:.0f}" if a else "—"

        def fmt_fails(a: AgentMetrics | None) -> str:
            return f"{a.num_parse_failures}" if a else "—"

        lines.append(f"| Inference Speed | {fmt_lat(ht)} | {fmt_lat(hr)} | {fmt_lat(pz)} |")
        lines.append(f"| Accuracy (in-dist) | {fmt_acc(ht)} | {fmt_acc(hr)} | {fmt_acc(pz)} |")
        lines.append(f"| Explainability | Low (softmax only) | Low (softmax only) | High (CoT text) |")
        lines.append(f"| Variable Choices | No (fixed head size) | No (fixed head size) | Yes (prompt) |")
        lines.append(f"| Pre-training Required | Yes (head MLP) | No | No (zero-shot) |")
        lines.append(f"| Parse Failures | 0 | 0 | {fmt_fails(pz)} |")
        lines.append(f"| Avg Tokens Generated | 0 | 0 | {fmt_toks(pz)} |")

    # Heldout comparison
    if "test_heldout" in all_metrics:
        lines.append("")
        lines.append("## Generalization (held-out layouts)")
        lines.append("")
        h = all_metrics["test_heldout"]
        ht_h = h.get("head_trained")
        pz_h = h.get("prompt_lm_zero_shot")
        ht_i = all_metrics.get("test_indist", {}).get("head_trained")
        pz_i = all_metrics.get("test_indist", {}).get("prompt_lm_zero_shot")

        lines.append("| Metric | Head (trained, indist→heldout) | Prompt LM (indist→heldout) |")
        lines.append("|--------|-------------------------------|---------------------------|")

        def delta(new: AgentMetrics | None, old: AgentMetrics | None) -> str:
            if new is None or old is None:
                return "—"
            diff = (new.accuracy - old.accuracy) * 100
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:.1f}pp"

        lines.append(f"| Accuracy Drop | {fmt_acc(ht_h)} vs {fmt_acc(ht_i)} ({delta(ht_h, ht_i)}) "
                     f"| {fmt_acc(pz_h)} vs {fmt_acc(pz_i)} ({delta(pz_h, pz_i)}) |")

        lines.append("")
        lines.append("**Observations:**")
        lines.append(f"- Head (trained) accuracy delta on heldout: {delta(ht_h, ht_i)}")
        lines.append(f"- Prompt LM accuracy delta on heldout: {delta(pz_h, pz_i)}")

    # Sample outputs from prompt LM for qualitative analysis
    if ref_test in all_metrics:
        m = all_metrics[ref_test]
        pz = m.get("prompt_lm_zero_shot")
        if pz and pz.raw_outputs:
            lines.append("")
            lines.append("## Sample Prompt LM Outputs")
            lines.append("")
            for i, out in enumerate(pz.raw_outputs[:5]):
                lines.append(f"### Example {i+1}")
                lines.append("```")
                lines.append(out)
                lines.append("```")
                lines.append("")

    return lines