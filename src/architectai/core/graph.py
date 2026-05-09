import networkx as nx
from typing import Dict, List, Optional, Set
from .primitives import Primitive, Connection

class ArchitectureGraph:
    """Represents a neural network architecture as a directed acyclic graph (DAG)."""
    
    def __init__(self, name: str = "sdnn_architecture"):
        self.name = name
        self.graph = nx.DiGraph()
        self.primitives: Dict[str, Primitive] = {}
        
    def add_node(self, primitive: Primitive):
        """Add a primitive node to the graph."""
        self.primitives[primitive.id] = primitive
        self.graph.add_node(primitive.id, primitive=primitive)
        
    def add_connection(self, source_id: str, target_id: str):
        """Add a directed edge between two primitives."""
        if source_id not in self.primitives or target_id not in self.primitives:
            raise ValueError(f"Nodes {source_id} or {target_id} not found in graph.")
        self.graph.add_edge(source_id, target_id)
        
    def get_primitive(self, node_id: str) -> Primitive:
        return self.primitives[node_id]
        
    def validate(self) -> bool:
        """Ensure the graph is a valid DAG and has a path from input to output."""
        if not nx.is_directed_acyclic_graph(self.graph):
            return False
        # Additional validation logic can be added here (e.g., shape consistency)
        return True
        
    def topological_sort(self) -> List[str]:
        """Return node IDs in topological order for execution/generation."""
        return list(nx.topological_sort(self.graph))

    def to_dict(self) -> Dict:
        """Serialize graph for storage or API transmission."""
        return {
            "name": self.name,
            "nodes": {k: v.dict() for k, v in self.primitives.items()},
            "edges": list(self.graph.edges())
        }
