import pytest
from architectai.core.dsl import ArchitectureDSL
from architectai.core.graph import ArchitectureGraph
from architectai.core.primitives import OperationType

def test_dsl_construction():
    dsl = ArchitectureDSL("test_model")
    graph = (
        dsl.input([3, 32, 32])
        .conv2d(16)
        .relu()
        .linear(10)
        .build()
    )
    
    assert isinstance(graph, ArchitectureGraph)
    assert len(graph.topological_sort()) == 4
    
    # Check node ops
    nodes = graph.topological_sort()
    assert graph.get_primitive(nodes[0]).config.op == OperationType.IDENTITY
    assert graph.get_primitive(nodes[1]).config.op == OperationType.CONV2D
    assert graph.get_primitive(nodes[2]).config.op == OperationType.RELU
    assert graph.get_primitive(nodes[3]).config.op == OperationType.LINEAR

def test_dsl_shape_inheritance():
    dsl = ArchitectureDSL("shape_test")
    graph = dsl.input([3, 32, 32]).build()
    
    input_node_id = graph.topological_sort()[0]
    input_node = graph.get_primitive(input_node_id)
    assert input_node.output_shape == [3, 32, 32]
