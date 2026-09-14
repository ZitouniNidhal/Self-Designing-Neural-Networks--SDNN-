from typing import Dict, List, Optional


def plot_evolution(history: List[Dict[str, float]], save_path: Optional[str] = None) -> str:
    """Plot evolution history (max, mean, min fitness over generations)."""
    if not history:
        return "No evolution history provided."

    generations = list(range(1, len(history) + 1))
    max_scores = [h.get("max", h.get("best", 0.0)) for h in history]
    mean_scores = [h.get("mean", 0.0) for h in history]

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(generations, max_scores, label="Best Fitness", color="green", marker="o")
        ax.plot(generations, mean_scores, label="Mean Fitness", color="blue", linestyle="--")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness Score")
        ax.set_title("Evolution Progress")
        ax.legend()
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            return f"Evolution plot saved to {save_path}"
        plt.close(fig)
        return "Evolution plot created successfully"
    except Exception:
        summary = [f"Generation {i+1}: Best = {max_scores[i]:.4f}" for i in range(len(history))]
        return "\n".join(summary)
