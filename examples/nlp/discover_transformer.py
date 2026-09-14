from pathlib import Path
from typing import Optional

from architectai import Architect
from architectai.core.dsl import ArchitectureDSL
from architectai.hardware.constraints import HardwareConstraints


def build_sample_architecture():
    dsl = ArchitectureDSL("transformer_explorer")
    return dsl.input([128]).linear(128).relu().linear(64).build()


def main(output_path: Optional[str] = None):
    output_path = output_path or "experiments/results/transformer_example.py"
    constraints = HardwareConstraints(max_params=600_000, max_memory_mb=256)
    architect = Architect()
    graph = architect.discover(task="nlp", iterations=8, constraints=constraints)

    if graph is None:
        graph = build_sample_architecture()

    architect.export(graph, output_path)
    print(f"Exported NLP example architecture to {Path(output_path).resolve()}")
    return graph


if __name__ == "__main__":
    main()
