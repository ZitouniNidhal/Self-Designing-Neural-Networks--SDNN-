from pathlib import Path
from typing import Optional

from architectai import Architect
from architectai.core.dsl import ArchitectureDSL
from architectai.hardware.constraints import HardwareConstraints


def build_sample_architecture():
    dsl = ArchitectureDSL("vision_language_explorer")
    return dsl.input([3, 224, 224]).conv2d(32).relu().conv2d(64).relu().linear(128).build()


def main(output_path: Optional[str] = None):
    output_path = output_path or "experiments/results/vision_language_example.py"
    constraints = HardwareConstraints(max_params=700_000, max_memory_mb=384)
    architect = Architect()
    graph = architect.discover(task="multimodal", iterations=10, constraints=constraints)

    if graph is None:
        graph = build_sample_architecture()

    architect.export(graph, output_path)
    print(f"Exported multimodal example architecture to {Path(output_path).resolve()}")
    return graph


if __name__ == "__main__":
    main()
