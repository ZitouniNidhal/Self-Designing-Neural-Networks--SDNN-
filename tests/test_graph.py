from architectai.core.graph import ArchitectureGraph
from architectai.core.primitives import NodeConfig, OperationType, Primitive


def test_graph_connection():
    graph = ArchitectureGraph("graph_connection")

    node_a = Primitive(
        id="node_a",
        config=NodeConfig(op=OperationType.IDENTITY),
    )
    node_b = Primitive(
        id="node_b",
        config=NodeConfig(
            op=OperationType.LINEAR,
            params={"out_features": 10},
        ),
    )

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_connection("node_a", "node_b")

    assert graph.validate()
    assert tuple(graph.to_dict()["edges"]) == (("node_a", "node_b"),)
