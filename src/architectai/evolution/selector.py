from typing import List
from ..core.graph import ArchitectureGraph

class Selector:
    def select(self, candidates: List[ArchitectureGraph], k: int = 1) -> List[ArchitectureGraph]:
        return candidates[:k]
