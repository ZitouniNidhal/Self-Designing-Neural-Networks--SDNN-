from architectai.core.dsl import ArchitectureDSL
from architectai.evolution.crossover import Crossover
from architectai.evolution.mutator import Mutator
from architectai.evolution.population import Population
from architectai.evolution.selector import Selector


def sample_graph(name="sample"):
    return ArchitectureDSL(name).input([3, 32, 32]).conv2d(16).relu().linear(10).build()


def test_mutator():
    graph = sample_graph("mut_test")
    mutator = Mutator()
    mutated = mutator.mutate(graph)

    assert mutated is not None
    assert mutated.name != graph.name


def test_crossover():
    parent_a = sample_graph("parent_a")
    parent_b = sample_graph("parent_b")
    cross = Crossover()
    child = cross.crossover(parent_a, parent_b)

    assert child is not None
    assert len(child.primitives) > 0


def test_selector():
    g1 = sample_graph("g1")
    g2 = sample_graph("g2")
    selector = Selector(strategy="tournament")

    selected = selector.select([g1, g2], scores=[0.2, 0.8], k=1)
    assert len(selected) == 1


def test_population():
    pop = Population()
    g1 = sample_graph("g1")
    g2 = sample_graph("g2")

    pop.add(g1, 0.5)
    pop.add(g2, 0.9)

    assert len(pop) == 2
    assert pop.best().name == "g2"
    stats = pop.stats()
    assert stats["max"] == 0.9
