from typing import Any, Dict, Optional

from .base import BaseEvaluator
from .profiler import Profiler


class ImageClassificationEvaluator(BaseEvaluator):
    """Evaluates architectures on image classification tasks."""

    def __init__(self):
        self.profiler = Profiler()

    def evaluate(
        self, graph: Any, data: Optional[Any] = None
    ) -> Dict[str, float]:
        stats = self.profiler.profile(graph)
        params = stats["params"]

        # Synthetic/heuristic accuracy model based on architecture parameters & depth
        accuracy = min(0.95, 0.5 + (params / 1e6) * 0.4)
        loss = max(0.1, 2.0 - accuracy * 2.0)

        return {
            "accuracy": round(accuracy, 4),
            "loss": round(loss, 4),
            "params": float(params),
            "macs": float(stats["macs"]),
        }

    def summarize(self, results: Dict[str, float]) -> str:
        acc = results.get("accuracy", 0.0)
        loss = results.get("loss", 0.0)
        params = int(results.get("params", 0))
        return (
            f"Image Classification - Accuracy: {acc * 100:.2f}%, "
            f"Loss: {loss:.4f}, Params: {params:,}"
        )
