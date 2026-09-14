import copy
from typing import Optional

from ..core.graph import ArchitectureGraph
from ..core.primitives import NodeConfig, OperationType
from .device_profiles import get_device_profile


class EdgeOptimizer:
    """Optimizes architecture graphs for edge device constraints."""

    def __init__(self, target_device: str = "jetson_nano"):
        self.target_device = target_device
        self.profile = get_device_profile(target_device)

    def optimize(
        self,
        graph: ArchitectureGraph,
        max_filters: Optional[int] = None,
    ) -> ArchitectureGraph:
        """Apply graph transformations to satisfy edge hardware constraints."""
        optimized = copy.deepcopy(graph)
        optimized.name = f"{graph.name}_edge_opt"

        cap_filters = max_filters or (
            64 if self.profile["category"] == "edge" else 128
        )

        for nid in list(optimized.primitives.keys()):
            primitive = optimized.primitives[nid]
            op = primitive.config.op

            if op == OperationType.CONV2D:
                params = dict(primitive.config.params)

                # Cap filter count for edge memory savings
                if params.get("filters", 0) > cap_filters:
                    params["filters"] = cap_filters

                # Reduce 5x5 kernels to 3x3 on edge devices
                if params.get("kernel_size", 3) > 3:
                    params["kernel_size"] = 3

                primitive.config = NodeConfig(
                    op=OperationType.CONV2D, params=params
                )

        return optimized

