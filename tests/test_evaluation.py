from architectai.core.dsl import ArchitectureDSL
from architectai.evaluation.image_classification import (
    ImageClassificationEvaluator,
)
from architectai.evaluation.metrics import (
    compute_accuracy,
    compute_f1,
    compute_mae,
    compute_mse,
)
from architectai.evaluation.profiler import Profiler


def test_metrics():
    acc = compute_accuracy([1, 0, 1, 1], [1, 0, 0, 1])
    assert acc == 0.75

    f1 = compute_f1([1, 0, 1, 0], [1, 0, 1, 1])
    assert round(f1, 2) == 0.80

    mse = compute_mse([1.0, 2.0], [1.0, 4.0])
    assert mse == 2.0

    mae = compute_mae([1.0, 2.0], [1.0, 4.0])
    assert mae == 1.0


def test_profiler():
    dsl = ArchitectureDSL("prof_test")
    graph = dsl.input([3, 32, 32]).conv2d(16).linear(10).build()

    profiler = Profiler()
    stats = profiler.profile(graph)

    assert stats["params"] > 0
    assert stats["num_layers"] == 3


def test_image_classification_evaluator():
    dsl = ArchitectureDSL("eval_test")
    graph = dsl.input([3, 32, 32]).conv2d(32).relu().linear(10).build()

    evaluator = ImageClassificationEvaluator()
    results = evaluator.evaluate(graph)

    assert "accuracy" in results
    assert "loss" in results
    summary = evaluator.summarize(results)
    assert "Image Classification" in summary
