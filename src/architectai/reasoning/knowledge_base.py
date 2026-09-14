from typing import Any, Dict, List, Optional


class KnowledgeBase:
    """Stores architectural best practices, patterns, and design heuristics."""

    def __init__(self):
        self._patterns: Dict[str, Dict[str, Any]] = {
            "conv_bn_relu": {
                "sequence": ["conv2d", "batchnorm", "relu"],
                "score_bonus": 0.2,
                "description": "Standard Conv -> BatchNorm -> ReLU block",
            },
            "conv_relu": {
                "sequence": ["conv2d", "relu"],
                "score_bonus": 0.1,
                "description": "Conv -> ReLU block",
            },
            "double_pooling_antipattern": {
                "sequence": ["maxpool2d", "maxpool2d"],
                "score_penalty": 0.3,
                "description": "Consecutive pooling shrinks feature map rapidly",
            },
        }

    def query(self, key: str) -> Optional[Dict[str, Any]]:
        """Query a pattern or heuristic by key."""
        return self._patterns.get(key)

    def add_pattern(
        self, key: str, sequence: List[str], score_bonus: float, description: str
    ) -> None:
        """Register a new architectural design pattern."""
        self._patterns[key] = {
            "sequence": sequence,
            "score_bonus": score_bonus,
            "description": description,
        }

    def list_patterns(self) -> List[str]:
        """List all known pattern names."""
        return list(self._patterns.keys())
