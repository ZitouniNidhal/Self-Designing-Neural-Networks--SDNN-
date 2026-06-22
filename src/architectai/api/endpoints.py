from fastapi import APIRouter
from ..core.graph import ArchitectureGraph

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/export")
def export_architecture(graph: ArchitectureGraph):
    return graph.to_dict()
