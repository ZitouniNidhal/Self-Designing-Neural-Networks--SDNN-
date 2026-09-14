import copy
import random

from ..core.graph import ArchitectureGraph
from ..core.primitives import Primitive


class Crossover:
    """Performs crossover recombination between two architecture DAGs."""

    def crossover(
        self, parent_a: ArchitectureGraph, parent_b: ArchitectureGraph
    ) -> ArchitectureGraph:
        """Combine layer prefix from parent_a with layer suffix from parent_b."""
        child = ArchitectureGraph(
            name=f"crossover_{parent_a.name}_{parent_b.name}"
        )

        nodes_a = parent_a.topological_sort()
        nodes_b = parent_b.topological_sort()

        if not nodes_a or not nodes_b:
            return copy.deepcopy(parent_a)

        split_a = len(nodes_a) // 2
        split_b = len(nodes_b) // 2

        prefix_ids = nodes_a[:split_a]
        suffix_ids = nodes_b[split_b:]

        last_id = None
        for nid in prefix_ids:
            orig = parent_a.get_primitive(nid)
            new_node = Primitive(
                id=f"c_{orig.id}",
                config=orig.config,
                input_shape=orig.input_shape,
                output_shape=orig.output_shape,
            )
            child.add_node(new_node)
            if last_id:
                child.add_connection(last_id, new_node.id)
            last_id = new_node.id

        for nid in suffix_ids:
            orig = parent_b.get_primitive(nid)
            new_id = f"c_{orig.id}_{random.randint(100, 999)}"
            new_node = Primitive(
                id=new_id,
                config=orig.config,
                input_shape=orig.input_shape,
                output_shape=orig.output_shape,
            )
            child.add_node(new_node)
            if last_id:
                child.add_connection(last_id, new_id)
            last_id = new_id

        return child

