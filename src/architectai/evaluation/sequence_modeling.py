from .base import BaseEvaluator

class SequenceModelingEvaluator(BaseEvaluator):
    def evaluate(self, graph, data=None):
        return {"perplexity": 0.0}

    def summarize(self, results):
        return f"Sequence modeling results: {results}"
