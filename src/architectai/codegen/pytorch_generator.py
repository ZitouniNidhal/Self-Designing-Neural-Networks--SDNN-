from typing import List
from ..core.graph import ArchitectureGraph
from ..core.primitives import OperationType

class PyTorchGenerator:
    """Generate a PyTorch module from an architecture graph."""

    def __init__(self, graph: ArchitectureGraph):
        self.graph = graph

    def generate(self) -> str:
        sections: List[str] = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "\n\n",
            f"class {self.graph.name.title().replace('_', '')}(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
        ]

        for node_id in self.graph.topological_sort():
            primitive = self.graph.get_primitive(node_id)
            if primitive.config.op == OperationType.CONV2D:
                params = primitive.config.params
                filters = params.get("filters", 32)
                kernel = params.get("kernel_size", 3)
                stride = params.get("stride", 1)
                sections.append(
                    f"        self.{node_id} = nn.Conv2d(3, {filters}, kernel_size={kernel}, stride={stride}, padding={kernel // 2})"
                )
            elif primitive.config.op == OperationType.BATCHNORM:
                sections.append(f"        self.{node_id} = nn.BatchNorm2d(32)")
            elif primitive.config.op == OperationType.LINEAR:
                out_features = primitive.config.params.get("out_features", 10)
                sections.append(f"        self.{node_id} = nn.Linear(128, {out_features})")
            elif primitive.config.op == OperationType.MAXPOOL2D:
                sections.append(f"        self.{node_id} = nn.MaxPool2d(kernel_size=2)")

        sections.append("\n    def forward(self, x):")
        for node_id in self.graph.topological_sort():
            primitive = self.graph.get_primitive(node_id)
            op = primitive.config.op
            if op == OperationType.IDENTITY:
                sections.append("        x = x")
            elif op == OperationType.CONV2D:
                sections.append(f"        x = self.{node_id}(x)")
            elif op == OperationType.RELU:
                sections.append("        x = F.relu(x)")
            elif op == OperationType.BATCHNORM:
                sections.append(f"        x = self.{node_id}(x)")
            elif op == OperationType.MAXPOOL2D:
                sections.append(f"        x = self.{node_id}(x)")
            elif op == OperationType.LINEAR:
                sections.append("        x = torch.flatten(x, 1)")
                sections.append(f"        x = self.{node_id}(x)")

        sections.append("        return x")
        return "\n".join(sections)
