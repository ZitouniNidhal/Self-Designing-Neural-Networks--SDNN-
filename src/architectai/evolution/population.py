import statistics
from typing import Dict, List, Optional, Tuple

from ..core.graph import ArchitectureGraph


class Population:
    """Manages a population of architecture graphs and their fitness scores."""

    def __init__(
        self,
        individuals: Optional[List[ArchitectureGraph]] = None,
        max_size: int = 100,
    ):
        self.individuals: List[ArchitectureGraph] = individuals or []
        self.scores: Dict[str, float] = {}
        self.max_size = max_size

    def add(self, graph: ArchitectureGraph, score: float = 0.0) -> None:
        """Add individual and store its fitness score."""
        self.individuals.append(graph)
        self.scores[graph.name] = score
        if len(self.individuals) > self.max_size:
            self._prune()

    def set_score(self, graph: ArchitectureGraph, score: float) -> None:
        """Set or update the fitness score for an individual."""
        self.scores[graph.name] = score

    def get_score(self, graph: ArchitectureGraph) -> float:
        """Get fitness score for an individual."""
        return self.scores.get(graph.name, 0.0)

    def best(self) -> Optional[ArchitectureGraph]:
        """Return the highest fitness individual."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda g: self.scores.get(g.name, 0.0))

    def get_top_k(self, k: int = 5) -> List[ArchitectureGraph]:
        """Return top-k individuals sorted by fitness descending."""
        sorted_pop = sorted(
            self.individuals,
            key=lambda g: self.scores.get(g.name, 0.0),
            reverse=True,
        )
        return sorted_pop[:k]

    def stats(self) -> Dict[str, float]:
        """Return population fitness statistics (mean, std, max, min)."""
        if not self.scores:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
        vals = list(self.scores.values())
        return {
            "mean": float(statistics.mean(vals)),
            "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "max": float(max(vals)),
            "min": float(min(vals)),
        }

    def _prune(self) -> None:
        """Remove lowest scoring individuals when population exceeds max_size."""
        sorted_pop = sorted(
            self.individuals,
            key=lambda g: self.scores.get(g.name, 0.0),
            reverse=True,
        )
        self.individuals = sorted_pop[: self.max_size]
        valid_names = {g.name for g in self.individuals}
        self.scores = {
            k: v for k, v in self.scores.items() if k in valid_names
        }

    def __len__(self) -> int:
        return len(self.individuals)

