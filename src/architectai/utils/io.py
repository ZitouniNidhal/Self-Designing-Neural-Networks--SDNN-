import json
from pathlib import Path


def read_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
