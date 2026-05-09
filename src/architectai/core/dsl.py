from typing import List, Optional
from .graph import ArchitectureGraph
from .primitives import Primitive, NodeConfig, OperationType

class ArchitectureDSL:
    """A high-level DSL for defining and manipulating SDNN architectures."""
    
    def __init__(self, name: str = "custom_model"):
        self.graph = ArchitectureGraph(name)
        self._node_counter = 0
        self._last_node_id = None
        
    def _generate_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def input(self, shape: List[int]) -> "ArchitectureDSL":
        node_id = self._generate_id("input")
        primitive = Primitive(
            id=node_id,
            config=NodeConfig(op=OperationType.IDENTITY),
            output_shape=shape
        )
        self.graph.add_node(primitive)
        self._last_node_id = node_id
        return self

    def conv2d(self, filters: int, kernel_size: int = 3, stride: int = 1) -> "ArchitectureDSL":
        node_id = self._generate_id("conv2d")
        primitive = Primitive(
            id=node_id,
            config=NodeConfig(
                op=OperationType.CONV2D,
                params={"filters": filters, "kernel_size": kernel_size, "stride": stride}
            )
        )
        self.graph.add_node(primitive)
        if self._last_node_id:
            self.graph.add_connection(self._last_node_id, node_id)
        self._last_node_id = node_id
        return self

    def relu(self) -> "ArchitectureDSL":
        node_id = self._generate_id("relu")
        primitive = Primitive(id=node_id, config=NodeConfig(op=OperationType.RELU))
        self.graph.add_node(primitive)
        if self._last_node_id:
            self.graph.add_connection(self._last_node_id, node_id)
        self._last_node_id = node_id
        return self

    def linear(self, out_features: int) -> "ArchitectureDSL":
        node_id = self._generate_id("linear")
        primitive = Primitive(
            id=node_id,
            config=NodeConfig(
                op=OperationType.LINEAR,
                params={"out_features": out_features}
            )
        )
        self.graph.add_node(primitive)
        if self._last_node_id:
            self.graph.add_connection(self._last_node_id, node_id)
        self._last_node_id = node_id
        return self

    def build(self) -> ArchitectureGraph:
        if not self.graph.validate():
            raise ValueError("Invalid architecture graph.")
        return self.graph
