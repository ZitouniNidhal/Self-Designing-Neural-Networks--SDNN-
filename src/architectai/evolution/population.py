from typing import List, Optional

from ..core.graph import ArchitectureGraph


class Population:
    def __init__(self, individuals: Optional[List[ArchitectureGraph]] = None):
        self.individuals = individuals or []

    def add(self, graph: ArchitectureGraph):
        self.individuals.append(graph)

    def best(self):
        return self.individuals[0] if self.individuals else None
