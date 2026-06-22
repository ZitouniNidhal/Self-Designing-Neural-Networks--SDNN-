from .base import BaseEvaluator

class ImageClassificationEvaluator(BaseEvaluator):
    def evaluate(self, graph, data=None):
        return {"accuracy": 0.0, "loss": 0.0}

    def summarize(self, results):
        return f"Image classification results: {results}"
