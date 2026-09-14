"""Run benchmark experiments across YAML configuration files."""

from pathlib import Path
from typing import Any, Dict

from architectai import Architect
from architectai.hardware.constraints import HardwareConstraints


def run_single_benchmark(config_path: str) -> Dict[str, Any]:
    """Execute architecture search benchmark for a single config file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config path {config_path} does not exist.")
        return {}

    print(f"Running benchmark for config: {path.name}")
    architect = Architect()

    constraints = HardwareConstraints(max_params=1_000_000, max_memory_mb=512)
    graph = architect.discover(
        task="image_classification",
        iterations=5,
        constraints=constraints,
    )

    out_file = Path("experiments/results") / f"bench_{path.stem}.py"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if graph:
        architect.export(graph, str(out_file))

    return {
        "config": path.name,
        "exported": str(out_file),
        "status": "success" if graph else "fallback",
    }


def main():
    config_dir = Path("experiments/configs")
    configs = list(config_dir.glob("*.yaml"))

    print(f"Found {len(configs)} benchmark configurations.")
    for cfg in configs:
        res = run_single_benchmark(str(cfg))
        print(f"Completed {cfg.name}: {res}")


if __name__ == "__main__":
    main()
