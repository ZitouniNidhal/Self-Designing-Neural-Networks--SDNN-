from pydantic import BaseModel
from typing import Any, Dict, List

class ArchitectureSchema(BaseModel):
    name: str
    nodes: Dict[str, Any]
    edges: List[List[str]]
