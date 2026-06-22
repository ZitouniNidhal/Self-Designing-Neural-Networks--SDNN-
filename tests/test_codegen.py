import tempfile
from architectai.codegen.compiler import Compiler
from architectai.core.dsl import ArchitectureDSL


def test_compiler_writes_json_and_py(tmp_path):
    dsl = ArchitectureDSL("compiler_test")
    graph = dsl.input([3, 32, 32]).conv2d(16).relu().linear(10).build()
    compiler = Compiler()

    json_path = tmp_path / "model.json"
    compiler.compile_to_file(graph, str(json_path))
    assert json_path.exists()
    assert json_path.read_text().startswith("{")

    py_path = tmp_path / "model.py"
    compiler.compile_to_file(graph, str(py_path))
    content = py_path.read_text()
    assert "class CompilerTest" in content or "class" in content
