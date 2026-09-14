from typing import Any, Sequence


def compute_accuracy(predictions: Sequence[Any], targets: Sequence[Any]) -> float:
    """Compute classification accuracy (proportion of correct predictions)."""
    if not predictions or len(predictions) != len(targets):
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def compute_f1(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """Compute binary F1 score."""
    if not predictions or len(predictions) != len(targets):
        return 0.0

    tp = sum(1 for p, t in zip(predictions, targets) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(predictions, targets) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, targets) if p == 0 and t == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_top_k_accuracy(
    predictions_prob: Sequence[Sequence[float]],
    targets: Sequence[int],
    k: int = 5,
) -> float:
    """Compute top-k classification accuracy."""
    if not predictions_prob or len(predictions_prob) != len(targets):
        return 0.0

    correct = 0
    for prob, t in zip(predictions_prob, targets):
        top_k_indices = sorted(range(len(prob)), key=lambda i: prob[i], reverse=True)[:k]
        if t in top_k_indices:
            correct += 1

    return correct / len(predictions_prob)


def compute_mse(predictions: Sequence[float], targets: Sequence[float]) -> float:
    """Compute Mean Squared Error."""
    if not predictions or len(predictions) != len(targets):
        return 0.0
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)


def compute_mae(predictions: Sequence[float], targets: Sequence[float]) -> float:
    """Compute Mean Absolute Error."""
    if not predictions or len(predictions) != len(targets):
        return 0.0
    return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)
