from typing import Optional

from ..core.graph import ArchitectureGraph
from .engine import ReasoningEngine


class Explainer:
    """Generates natural language explanations of architecture design choices."""

    def __init__(self, engine: Optional[ReasoningEngine] = None):
        self.engine = engine or ReasoningEngine()

    def explain(self, graph: ArchitectureGraph) -> str:
        """Produce a human-readable diagnosis and summary of the architecture."""
        res = self.engine.validate(graph)
        lines = [f"Architecture Analysis for '{graph.name}':"]

        num_nodes = len(graph.primitives)
        num_edges = graph.graph.number_of_edges()
        lines.append(f"- Nodes: {num_nodes}, Connections: {num_edges}")

        if res["is_valid"]:
            lines.append("- Status: VALID DAG architecture")
        else:
            lines.append("- Status: INVALID architecture")
            lines.append("  Violations found:")
            for v in res["violations"]:
                lines.append(f"    * [{v['rule']}]: {v['message']}")

        pattern_score = res["pattern_score"]
        lines.append(f"- Pattern Adherence Score: {pattern_score:.2f} / 1.00")

        ops = [n.config.op for n in graph.primitives.values()]
        lines.append(f"- Layer Composition: {len(ops)} operations ({set(ops)})")

        return "\n".join(lines)
