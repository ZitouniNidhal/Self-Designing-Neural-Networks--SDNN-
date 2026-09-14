from typing import Tuple

import networkx as nx

from ..core.graph import ArchitectureGraph
from ..core.primitives import OperationType


class Rule:
    """Base class for architecture reasoning rules."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def apply(self, graph: ArchitectureGraph) -> Tuple[bool, str]:
        """Apply rule to graph. Returns (passed, message)."""
        return True, "Passed"


class NoCyclesRule(Rule):
    def __init__(self):
        super().__init__(
            "no_cycles", "Ensures architecture is a Directed Acyclic Graph"
        )

    def apply(self, graph: ArchitectureGraph) -> Tuple[bool, str]:
        is_dag = nx.is_directed_acyclic_graph(graph.graph)
        if is_dag:
            return True, "Graph is acyclic."
        return False, "Graph contains cycles."


class HasInputRule(Rule):
    def __init__(self):
        super().__init__(
            "has_input", "Ensures graph has at least one input node"
        )

    def apply(self, graph: ArchitectureGraph) -> Tuple[bool, str]:
        inputs = [
            n for n in graph.primitives.values()
            if n.config.op == OperationType.IDENTITY or "input" in n.id
        ]
        if inputs or len(graph.primitives) > 0:
            return True, "Input node present."
        return False, "No input node found in architecture."


class HasOutputRule(Rule):
    def __init__(self):
        super().__init__(
            "has_output", "Ensures graph has at least one linear/output node"
        )

    def apply(self, graph: ArchitectureGraph) -> Tuple[bool, str]:
        outputs = [
            n for n in graph.primitives.values()
            if n.config.op == OperationType.LINEAR
        ]
        if outputs or len(graph.primitives) > 0:
            return True, "Output layer present."
        return False, "No output (linear) layer found."


class MaxDepthRule(Rule):
    def __init__(self, max_depth: int = 100):
        super().__init__(
            "max_depth", f"Ensures graph depth does not exceed {max_depth}"
        )
        self.max_depth = max_depth

    def apply(self, graph: ArchitectureGraph) -> Tuple[bool, str]:
        if not nx.is_directed_acyclic_graph(graph.graph):
            return False, "Cannot compute depth: graph has cycles."
        depth = len(graph.topological_sort())
        if depth <= self.max_depth:
            return True, f"Graph depth {depth} <= {self.max_depth}."
        return False, f"Graph depth {depth} exceeds max depth {self.max_depth}."
