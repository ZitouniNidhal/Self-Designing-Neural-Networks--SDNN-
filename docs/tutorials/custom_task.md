# Custom Task Definition Tutorial

Learn how to define custom task evaluators and register domain-specific search spaces.

## 1. Implement a Custom Evaluator

Inherit from `BaseEvaluator`:

```python
from architectai.evaluation.base import BaseEvaluator

class CustomMedicalEvaluator(BaseEvaluator):
    def evaluate(self, graph, data=None):
        # Profile model parameters & compute domain score
        num_layers = len(graph.primitives)
        score = min(0.99, 0.70 + num_layers * 0.05)
        return {"dice_score": score, "loss": 1.0 - score}

    def summarize(self, results):
        return f"Medical Segmentation - Dice Score: {results['dice_score']:.4f}"
```

## 2. Registering with Architect

```python
from architectai import Architect

architect = Architect()
architect.register_evaluator("medical_segmentation", CustomMedicalEvaluator())

graph = architect.discover(task="medical_segmentation", iterations=5)
```
