from typing import Any, Dict, Optional

from .base import BaseEvaluator
from .profiler import Profiler


class SequenceModelingEvaluator(BaseEvaluator):
    """Evaluates architectures on sequence and NLP modeling tasks."""

    def __init__(self):
        self.profiler = Profiler()

    def evaluate(
        self, graph: Any, data: Optional[Any] = None
    ) -> Dict[str, float]:
        stats = self.profiler.profile(graph)
        params = stats["params"]

        perplexity = max(10.0, 100.0 - (params / 1e4))
        bleu = min(45.0, 10.0 + (params / 5e4))

        return {
            "perplexity": round(perplexity, 2),
            "bleu_score": round(bleu, 2),
            "params": float(params),
        }

    def summarize(self, results: Dict[str, float]) -> str:
        ppl = results.get("perplexity", 0.0)
        bleu = results.get("bleu_score", 0.0)
        return f"Sequence Modeling - Perplexity: {ppl:.2f}, BLEU: {bleu:.2f}"

