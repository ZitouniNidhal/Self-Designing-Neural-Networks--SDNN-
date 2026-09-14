import copy
import random
from typing import Optional

from ..core.graph import ArchitectureGraph
from ..core.primitives import NodeConfig, OperationType, Primitive


class Mutator:
    """Applies stochastic mutations to architecture DAGs."""

    def __init__(
        self,
        mutation_rate: float = 0.3,
        filter_choices: Optional[list] = None,
        kernel_choices: Optional[list] = None,
    ):
        self.mutation_rate = mutation_rate
        self.filter_choices = filter_choices or [16, 32, 64, 128]
        self.kernel_choices = kernel_choices or [3, 5]

    def mutate(self, graph: ArchitectureGraph) -> ArchitectureGraph:
        """Apply random mutations to the graph and return a new instance."""
        mutated = copy.deepcopy(graph)
        mutated.name = f"{graph.name}_mutated"
        nodes = list(mutated.primitives.values())
        if not nodes:
            return mutated

        mutation_type = random.choice(["param", "op_swap", "insert"])

        if mutation_type == "param":
            self._mutate_params(mutated)
        elif mutation_type == "op_swap":
            self._mutate_op_swap(mutated)
        elif mutation_type == "insert":
            self._mutate_insert(mutated)

        return mutated

    def _mutate_params(self, graph: ArchitectureGraph) -> None:
        conv_nodes = [n for n in graph.primitives.values() if n.config.op == OperationType.CONV2D]
        if conv_nodes:
            target = random.choice(conv_nodes)
            params = dict(target.config.params)
            params["filters"] = random.choice(self.filter_choices)
            params["kernel_size"] = random.choice(self.kernel_choices)
            target.config = NodeConfig(op=OperationType.CONV2D, params=params)

    def _mutate_op_swap(self, graph: ArchitectureGraph) -> None:
        swappable = [
            n
            for n in graph.primitives.values()
            if n.config.op in (OperationType.RELU, OperationType.BATCHNORM)
        ]
        if swappable:
            target = random.choice(swappable)
            new_op = (
                OperationType.BATCHNORM
                if target.config.op == OperationType.RELU
                else OperationType.RELU
            )
            target.config = NodeConfig(op=new_op)

    def _mutate_insert(self, graph: ArchitectureGraph) -> None:
        order = graph.topological_sort()
        if len(order) < 2:
            return
        idx = random.randint(0, len(order) - 2)
        src_id = order[idx]
        dst_id = order[idx + 1]

        if graph.graph.has_edge(src_id, dst_id):
            graph.graph.remove_edge(src_id, dst_id)
            new_id = f"inserted_{random.randint(1000, 9999)}"
            new_node = Primitive(
                id=new_id,
                config=NodeConfig(op=OperationType.RELU),
            )
            graph.add_node(new_node)
            graph.add_connection(src_id, new_id)
            graph.add_connection(new_id, dst_id)
