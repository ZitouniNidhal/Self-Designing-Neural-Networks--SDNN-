from pathlib import Path
from typing import Optional

from architectai import Architect
from architectai.core.dsl import ArchitectureDSL
from architectai.hardware.constraints import HardwareConstraints


def build_imagenet_architecture():
    dsl = ArchitectureDSL("imagenet_backbone")
    return (
        dsl.input([3, 224, 224])
        .conv2d(64, kernel_size=7, stride=2)
        .relu()
        .conv2d(128, kernel_size=3)
        .relu()
        .conv2d(256, kernel_size=3)
        .relu()
        .linear(1000)
        .build()
    )


def main(output_path: Optional[str] = None):
    output_path = (
        output_path or "experiments/results/imagenet_example.py"
    )
    constraints = HardwareConstraints(
        max_params=25_000_000, max_memory_mb=4096
    )
    architect = Architect()
    graph = architect.discover(
        task="image_classification", iterations=10, constraints=constraints
    )

    if graph is None:
        graph = build_imagenet_architecture()

    architect.export(graph, output_path)
    print(f"Exported ImageNet architecture to {Path(output_path).resolve()}")
    return graph


if __name__ == "__main__":
    main()
