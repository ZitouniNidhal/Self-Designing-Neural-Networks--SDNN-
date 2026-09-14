from pathlib import Path
from typing import Optional

from architectai import Architect
from architectai.core.dsl import ArchitectureDSL
from architectai.hardware.constraints import HardwareConstraints


def build_sample_architecture():
    dsl = ArchitectureDSL("mobile_net_explorer")
    return dsl.input([3, 224, 224]).conv2d(16).relu().conv2d(32).relu().linear(10).build()


def main(output_path: Optional[str] = None):
    output_path = output_path or "experiments/results/mobile_net_example.py"
    constraints = HardwareConstraints(max_params=300_000, max_memory_mb=128)
    architect = Architect()
    graph = architect.discover(task="edge", iterations=6, constraints=constraints)

    if graph is None:
        graph = build_sample_architecture()

    architect.export(graph, output_path)
    print(f"Exported edge example architecture to {Path(output_path).resolve()}")
    return graph


if __name__ == "__main__":
    main()
