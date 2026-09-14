"""Analyze experiment result files and print summary reports."""

from pathlib import Path
from typing import Any, Dict, List


def analyze_results(results_dir: str = "experiments/results") -> Dict[str, Any]:
    """Inspect and report on generated experiment files."""
    path = Path(results_dir)
    if not path.exists():
        return {"total_files": 0, "files": [], "status": "no results directory"}

    files: List[str] = [
        str(p.name) for p in path.glob("**/*") if p.is_file()
    ]
    total_size = sum(p.stat().st_size for p in path.glob("**/*") if p.is_file())

    report = {
        "results_directory": str(path.resolve()),
        "total_files": len(files),
        "total_bytes": total_size,
        "files": files,
    }

    return report


def main():
    report = analyze_results()
    print("ArchitectAI Experiment Results Summary:")
    print(f"- Directory: {report.get('results_directory')}")
    print(f"- Total Files: {report.get('total_files')}")
    print(f"- Total Size: {report.get('total_bytes')} bytes")
    for f in report.get("files", []):
        print(f"  * {f}")


if __name__ == "__main__":
    main()
