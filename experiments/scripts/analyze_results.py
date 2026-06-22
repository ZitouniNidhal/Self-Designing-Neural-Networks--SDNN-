"""Analyze experiment result files."""

from pathlib import Path


def analyze_results(results_dir: str):
    path = Path(results_dir)
    return {"files": [str(p) for p in path.glob('**/*') if p.is_file()]}


if __name__ == '__main__':
    print('Analyze results script placeholder')
