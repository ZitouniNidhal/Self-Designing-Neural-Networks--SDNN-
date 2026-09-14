import copy
from typing import Any, Dict

from ..core.graph import ArchitectureGraph
from ..core.primitives import NodeConfig


class Quantizer:
    """Applies precision quantization annotations to architecture graphs."""

    def __init__(self, precision: str = "int8"):
        if precision not in ("int8", "fp16", "int4"):
            raise ValueError(f"Unsupported quantization precision: {precision}")
        self.precision = precision

    def quantize(self, graph: ArchitectureGraph) -> ArchitectureGraph:
        """Annotate all weight-bearing primitives with target precision."""
        quantized = copy.deepcopy(graph)
        quantized.name = f"{graph.name}_{self.precision}"

        for nid in list(quantized.primitives.keys()):
            primitive = quantized.primitives[nid]
            params = dict(primitive.config.params)
            params["precision"] = self.precision
            primitive.config = NodeConfig(
                op=primitive.config.op, params=params
            )

        return quantized

    def estimate_compression(self, original_size_mb: float) -> Dict[str, Any]:
        """Estimate model size reduction following quantization."""
        ratio = 4.0 if self.precision == "int8" else 2.0
        if self.precision == "int4":
            ratio = 8.0
        quantized_mb = original_size_mb / ratio
        return {
            "precision": self.precision,
            "original_mb": round(original_size_mb, 2),
            "quantized_mb": round(quantized_mb, 2),
            "compression_ratio": ratio,
        }
