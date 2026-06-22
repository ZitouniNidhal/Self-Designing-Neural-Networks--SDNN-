from .base import BaseEvaluator

class MultimodalEvaluator(BaseEvaluator):
    def evaluate(self, graph, data=None):
        return {"multimodal_score": 0.0}

    def summarize(self, results):
        return f"Multimodal results: {results}"
