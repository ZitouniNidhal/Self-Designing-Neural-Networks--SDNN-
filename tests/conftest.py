import pytest
from architectai.core.dsl import ArchitectureDSL

@pytest.fixture
def sample_graph():
    dsl = ArchitectureDSL("fixture_model")
    return dsl.input([3, 32, 32]).conv2d(16).relu().linear(10).build()
