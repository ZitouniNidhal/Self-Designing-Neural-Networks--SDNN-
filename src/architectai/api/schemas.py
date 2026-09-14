from typing import Any, Dict, List

from pydantic import BaseModel


class ArchitectureSchema(BaseModel):
    name: str
    nodes: Dict[str, Any]
    edges: List[List[str]]
