from typing import Any, Dict, Optional


def generate_report(
    results: Dict[str, Any], save_path: Optional[str] = None
) -> str:
    """Generate a Markdown report summarizing an architectural discovery run."""
    task = results.get("task", "image_classification")
    generations = results.get("generations", 0)
    best_score = results.get("best_score", 0.0)
    best_arch = results.get("best_arch", "Unknown")
    stats = results.get("stats", {})

    lines = [
        "# ArchitectAI Discovery Summary Report",
        "",
        "## Search Overview",
        f"- **Task**: `{task}`",
        f"- **Generations Completed**: {generations}",
        f"- **Best Fitness Score**: {best_score:.4f}",
        f"- **Best Architecture Name**: `{best_arch}`",
        "",
        "## Performance Metrics",
    ]

    for k, v in stats.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")
        else:
            lines.append(f"- **{k}**: {v}")

    lines.extend(
        [
            "",
            "## Recommendations",
            "- Export PyTorch code using `Architect.export(best_model, 'pytorch')`",
            "- Apply INT8 quantization for edge deployment if targeting mobile/microcontrollers.",
            "",
            "---",
            "*Report generated automatically by ArchitectAI Viz Subsystem*",
        ]
    )

    report_content = "\n".join(lines)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        return f"Report saved to {save_path}"

    return report_content

