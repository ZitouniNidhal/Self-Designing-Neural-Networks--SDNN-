from architectai.core.dsl import ArchitectureDSL
from architectai.reasoning.engine import ReasoningEngine
from architectai.reasoning.explainer import Explainer
from architectai.reasoning.knowledge_base import KnowledgeBase
from architectai.reasoning.rules import HasInputRule, NoCyclesRule


def sample_graph():
    return ArchitectureDSL("reas_test").input([3, 32, 32]).conv2d(16).relu().linear(10).build()


def test_rules():
    graph = sample_graph()
    rule_cycle = NoCyclesRule()
    passed, msg = rule_cycle.apply(graph)
    assert passed is True

    rule_input = HasInputRule()
    passed, msg = rule_input.apply(graph)
    assert passed is True


def test_knowledge_base():
    kb = KnowledgeBase()
    pattern = kb.query("conv_bn_relu")
    assert pattern is not None
    assert "sequence" in pattern


def test_reasoning_engine_and_explainer():
    graph = sample_graph()
    engine = ReasoningEngine()
    validation = engine.validate(graph)

    assert validation["is_valid"] is True
    assert validation["pattern_score"] >= 0.0

    explainer = Explainer(engine)
    explanation = explainer.explain(graph)
    assert "Architecture Analysis" in explanation
