from dataclasses import dataclass
from typing import Optional

@dataclass
class CLIArgs:
    task: str = "image_classification"
    iterations: int = 10
    output: str = "best_model.json"
    device: Optional[str] = None
