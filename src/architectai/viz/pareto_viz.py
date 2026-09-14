from typing import Any, Dict, List, Optional


def plot_pareto(
    records: List[Dict[str, Any]], save_path: Optional[str] = None
) -> str:
    """Scatter plot of multi-objective trade-offs (e.g. Accuracy vs Parameters)."""
    if not records:
        return "No pareto records provided."

    accuracies = [r.get("accuracy", r.get("score", 0.0)) for r in records]
    params = [r.get("params", r.get("cost", 0.0)) for r in records]
    names = [r.get("name", f"Arch_{i}") for i, r in enumerate(records)]

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(params, accuracies, color="purple", s=100, alpha=0.7)

        for i, txt in enumerate(names):
            ax.annotate(
                txt,
                (params[i], accuracies[i]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax.set_xlabel("Parameters")
        ax.set_ylabel("Accuracy / Score")
        ax.set_title("Pareto Frontier Trade-Off")
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            return f"Pareto plot saved to {save_path}"
        plt.close(fig)
        return "Pareto plot created successfully"
    except Exception:
        summary = [f"{n}: Params={p}, Acc={a:.4f}" for n, p, a in zip(names, params, accuracies)]
        return "\n".join(summary)
