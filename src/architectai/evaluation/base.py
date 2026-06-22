from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """Base class for dataset and task evaluators."""

    @abstractmethod
    def evaluate(self, graph, data=None):
        raise NotImplementedError

    @abstractmethod
    def summarize(self, results):
        raise NotImplementedError
