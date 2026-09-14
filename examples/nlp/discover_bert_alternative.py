from pathlib import Path
from typing import Optional

from architectai import Architect
from architectai.core.dsl import ArchitectureDSL
from architectai.hardware.constraints import HardwareConstraints


def build_bert_alternative_architecture():
    dsl = ArchitectureDSL("bert_alternative")
    return dsl.input([128, 768]).linear(768).relu().linear(768).relu().linear(2).build()


def main(output_path: Optional[str] = None):
    output_path = output_path or "experiments/results/bert_alternative_example.py"
    constraints = HardwareConstraints(max_params=50_000_000, max_memory_mb=2048)
    architect = Architect()
    graph = architect.discover(task="sequence_modeling", iterations=8, constraints=constraints)

    if graph is None:
        graph = build_bert_alternative_architecture()

    architect.export(graph, output_path)
    print(f"Exported BERT alternative model to {Path(output_path).resolve()}")
    return graph


if __name__ == "__main__":
    main()
