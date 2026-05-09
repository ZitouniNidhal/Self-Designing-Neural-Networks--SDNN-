from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

class OperationType(str, Enum):
    CONV2D = "conv2d"
    MAXPOOL2D = "maxpool2d"
    AVGPOOL2D = "avgpool2d"
    BATCHNORM = "batchnorm"
    RELU = "relu"
    LINEAR = "linear"
    IDENTITY = "identity"
    ADD = "add"
    CONCAT = "concat"

class NodeConfig(BaseModel):
    op: OperationType
    params: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True

class Connection(BaseModel):
    source_id: str
    target_id: str
    
class Primitive(BaseModel):
    """Base class for all architectural primitives."""
    id: str
    config: NodeConfig
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None

    def __hash__(self):
        return hash(self.id)
