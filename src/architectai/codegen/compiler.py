import json
from pathlib import Path
from ..core.graph import ArchitectureGraph
from .pytorch_generator import PyTorchGenerator

class Compiler:
    """Compiler for translating architecture graphs into export formats."""

    def compile_to_file(self, graph: ArchitectureGraph, path: str):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if path.endswith(".py"):
            code = PyTorchGenerator(graph).generate()
            target.write_text(code, encoding="utf-8")
        else:
            target.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")
