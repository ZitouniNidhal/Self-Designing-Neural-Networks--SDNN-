import time
from typing import Any, Dict, List, Optional


class SystemState:
    """Tracks global search session status, metrics, and state history."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.status = "idle"
        self.current_generation = 0
        self.max_generations = 0
        self.best_score = 0.0
        self.best_architecture_name: Optional[str] = None
        self.history: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self, max_generations: int = 10) -> None:
        self.status = "running"
        self.max_generations = max_generations
        self.start_time = time.time()

    def update(
        self, generation: int, best_score: float, arch_name: str, metrics: Dict[str, Any]
    ) -> None:
        self.current_generation = generation
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_architecture_name = arch_name

        record = {
            "generation": generation,
            "best_score": best_score,
            "arch_name": arch_name,
            "timestamp": time.time(),
            **metrics,
        }
        self.history.append(record)

    def complete(self) -> None:
        self.status = "completed"
        self.end_time = time.time()

    def fail(self, reason: str = "") -> None:
        self.status = "failed"
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        elapsed = 0.0
        if self.start_time:
            end = self.end_time or time.time()
            elapsed = round(end - self.start_time, 2)

        return {
            "session_id": self.session_id,
            "status": self.status,
            "current_generation": self.current_generation,
            "max_generations": self.max_generations,
            "best_score": self.best_score,
            "best_architecture_name": self.best_architecture_name,
            "elapsed_seconds": elapsed,
            "history_length": len(self.history),
        }

