from typing import Any, Dict

from ..core.graph import ArchitectureGraph
from ..core.primitives import OperationType


class Profiler:
    """Hardware and model parameter/FLOPs profiler."""

    def profile(self, graph: ArchitectureGraph) -> Dict[str, Any]:
        """Profile an architecture graph for parameters, MACs, and memory."""
        total_params = 0
        total_macs = 0

        # Estimate params based on nodes
        in_channels = 3
        spatial_dim = 32

        for nid in graph.topological_sort():
            primitive = graph.get_primitive(nid)
            op = primitive.config.op
            params_dict = primitive.config.params

            if op == OperationType.CONV2D:
                out_channels = params_dict.get("filters", 32)
                k_size = params_dict.get("kernel_size", 3)
                node_params = in_channels * out_channels * k_size * k_size
                node_macs = node_params * spatial_dim * spatial_dim
                total_params += node_params
                total_macs += node_macs
                in_channels = out_channels
            elif op == OperationType.LINEAR:
                out_features = params_dict.get("out_features", 10)
                node_params = in_channels * out_features
                node_macs = node_params
                total_params += node_params
                total_macs += node_macs
                in_channels = out_features
            elif op in (OperationType.MAXPOOL2D, OperationType.AVGPOOL2D):
                spatial_dim = max(1, spatial_dim // 2)

        memory_mb = (total_params * 4) / (1024 * 1024)

        return {
            "params": total_params,
            "macs": total_macs,
            "flops": total_macs * 2,
            "estimated_memory_mb": round(memory_mb, 4),
            "num_layers": len(graph.primitives),
        }
