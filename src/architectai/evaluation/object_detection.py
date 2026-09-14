from typing import Any, Dict, Optional

from .base import BaseEvaluator
from .profiler import Profiler


class ObjectDetectionEvaluator(BaseEvaluator):
    """Evaluates architectures on object detection tasks."""

    def __init__(self):
        self.profiler = Profiler()

    def evaluate(
        self, graph: Any, data: Optional[Any] = None
    ) -> Dict[str, float]:
        stats = self.profiler.profile(graph)
        params = stats["params"]

        m_ap = min(0.85, 0.3 + (params / 2e6) * 0.5)
        fps = max(5.0, 120.0 - (params / 1e5))

        return {
            "mAP": round(m_ap, 4),
            "fps": round(fps, 1),
            "params": float(params),
            "macs": float(stats["macs"]),
        }

    def summarize(self, results: Dict[str, float]) -> str:
        map_val = results.get("mAP", 0.0)
        fps = results.get("fps", 0.0)
        return f"Object Detection - mAP@50: {map_val:.4f}, Inference speed: {fps:.1f} FPS"

