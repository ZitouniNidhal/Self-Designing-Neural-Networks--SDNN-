from pathlib import Path
from typing import Optional

from architectai import Architect
from architectai.core.dsl import ArchitectureDSL
from architectai.hardware.constraints import HardwareConstraints


def build_tinyml_architecture():
    dsl = ArchitectureDSL("micro_net")
    return (
        dsl.input([1, 16, 16])
        .conv2d(8, kernel_size=3)
        .relu()
        .conv2d(16, kernel_size=3)
        .relu()
        .linear(5)
        .build()
    )


def main(output_path: Optional[str] = None):
    output_path = output_path or "experiments/results/tinyml_example.py"
    constraints = HardwareConstraints(max_params=50_000, max_memory_mb=16)
    architect = Architect()
    graph = architect.discover(task="image_classification", iterations=5, constraints=constraints)

    if graph is None:
        graph = build_tinyml_architecture()

    architect.export(graph, output_path)
    print(f"Exported TinyML architecture to {Path(output_path).resolve()}")
    return graph


if __name__ == "__main__":
    main()
