import random
from typing import List, Optional

from ..core.graph import ArchitectureGraph


class Selector:
    """Selects candidate architectures based on fitness scores."""

    def __init__(self, strategy: str = "tournament", tournament_size: int = 3):
        self.strategy = strategy
        self.tournament_size = tournament_size

    def select(
        self,
        candidates: List[ArchitectureGraph],
        scores: Optional[List[float]] = None,
        k: int = 1,
    ) -> List[ArchitectureGraph]:
        """Select k individuals from candidates based on fitness scores."""
        if not candidates:
            return []

        k = min(k, len(candidates))

        if scores is None or len(scores) != len(candidates):
            return random.sample(candidates, k)

        if self.strategy == "tournament":
            return self._tournament_select(candidates, scores, k)
        elif self.strategy == "roulette":
            return self._roulette_select(candidates, scores, k)
        else:
            # Default to top-k by score
            indexed = sorted(
                zip(candidates, scores), key=lambda x: x[1], reverse=True
            )
            return [cand for cand, _ in indexed[:k]]

    def _tournament_select(
        self,
        candidates: List[ArchitectureGraph],
        scores: List[float],
        k: int,
    ) -> List[ArchitectureGraph]:
        selected: List[ArchitectureGraph] = []
        n = len(candidates)
        t_size = min(self.tournament_size, n)

        for _ in range(k):
            indices = random.sample(range(n), t_size)
            best_idx = max(indices, key=lambda idx: scores[idx])
            selected.append(candidates[best_idx])

        return selected

    def _roulette_select(
        self,
        candidates: List[ArchitectureGraph],
        scores: List[float],
        k: int,
    ) -> List[ArchitectureGraph]:
        min_score = min(scores)
        shift = abs(min_score) + 1.0 if min_score <= 0 else 0.0
        adj_scores = [s + shift for s in scores]
        total = sum(adj_scores)

        if total == 0:
            return random.sample(candidates, k)

        probs = [s / total for s in adj_scores]
        return random.choices(candidates, weights=probs, k=k)
