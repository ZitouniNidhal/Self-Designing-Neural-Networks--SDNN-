from typing import Any, Dict, List, Optional

from ..core.graph import ArchitectureGraph
from .knowledge_base import KnowledgeBase
from .rules import (
    HasInputRule,
    HasOutputRule,
    MaxDepthRule,
    NoCyclesRule,
    Rule,
)


class ReasoningEngine:
    """Evaluates architecture graphs against rules and design patterns."""

    def __init__(
        self,
        rules: Optional[List[Rule]] = None,
        kb: Optional[KnowledgeBase] = None,
    ):
        self.rules = rules or [
            NoCyclesRule(),
            HasInputRule(),
            HasOutputRule(),
            MaxDepthRule(),
        ]
        self.kb = kb or KnowledgeBase()

    def infer(self, graph: ArchitectureGraph) -> List[Dict[str, Any]]:
        """Run inference rules against graph and return violation findings."""
        findings = []
        for rule in self.rules:
            passed, msg = rule.apply(graph)
            if not passed:
                findings.append(
                    {
                        "rule": rule.name,
                        "description": rule.description,
                        "passed": False,
                        "message": msg,
                    }
                )
        return findings

    def evaluate_patterns(self, graph: ArchitectureGraph) -> float:
        """Calculate pattern quality score based on knowledge base heuristics."""
        try:
            topo = graph.topological_sort()
        except Exception:
            return 0.0

        op_seq = [
            str(graph.get_primitive(nid).config.op).lower() for nid in topo
        ]
        score = 0.5  # Base score

        # Check for conv -> batchnorm -> relu
        for i in range(len(op_seq) - 2):
            sub = op_seq[i : i + 3]
            if sub == ["conv2d", "batchnorm", "relu"]:
                score += 0.2
            elif sub[:2] == ["conv2d", "relu"]:
                score += 0.1

        return min(1.0, max(0.0, score))

    def validate(self, graph: ArchitectureGraph) -> Dict[str, Any]:
        """Comprehensive architecture validation and quality score."""
        findings = self.infer(graph)
        is_valid = len(findings) == 0
        pattern_score = self.evaluate_patterns(graph)
        return {
            "is_valid": is_valid,
            "violations": findings,
            "pattern_score": pattern_score,
            "total_score": pattern_score if is_valid else 0.0,
        }

