from .base import BaseEvaluator

class ObjectDetectionEvaluator(BaseEvaluator):
    def evaluate(self, graph, data=None):
        return {"mAP": 0.0}

    def summarize(self, results):
        return f"Object detection results: {results}"
