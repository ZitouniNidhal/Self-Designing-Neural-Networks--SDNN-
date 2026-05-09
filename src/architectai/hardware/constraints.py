from pydantic import BaseModel
from typing import Optional
from ..core.graph import ArchitectureGraph

class HardwareConstraints(BaseModel):
    max_params: Optional[int] = None
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    target_device: str = "cpu"

class HardwareEvaluator:
    """Evaluates an architecture against hardware constraints."""
    
    def __init__(self, constraints: HardwareConstraints):
        self.constraints = constraints

    def estimate_params(self, graph: ArchitectureGraph) -> int:
        """Rough estimation of parameters."""
        total_params = 0
        for node_id in graph.topological_sort():
            primitive = graph.get_primitive(node_id)
            if primitive.config.op == "conv2d":
                # Very rough: filters * kernel^2 * in_channels
                # This needs proper shape tracking to be accurate
                total_params += primitive.config.params.get("filters", 32) * 9 * 3 
            elif primitive.config.op == "linear":
                total_params += primitive.config.params.get("out_features", 10) * 100
        return total_params

    def satisfies(self, graph: ArchitectureGraph) -> bool:
        """Check if the graph meets all constraints."""
        if self.constraints.max_params:
            if self.estimate_params(graph) > self.constraints.max_params:
                return False
        return True
