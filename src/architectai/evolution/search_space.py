import random
from typing import List, Optional
from ..core.dsl import ArchitectureDSL
from ..core.graph import ArchitectureGraph
from ..core.primitives import OperationType

class SearchSpace:
    """Defines the boundaries and operations for architecture search."""
    
    def __init__(self):
        self.available_ops = [
            OperationType.CONV2D,
            OperationType.MAXPOOL2D,
            OperationType.RELU,
            OperationType.BATCHNORM,
            OperationType.IDENTITY
        ]
        self.filters_options = [16, 32, 64, 128]
        self.kernel_options = [3, 5]

    def sample_random_architecture(self, depth: int = 5) -> ArchitectureGraph:
        """Sample a random DAG architecture."""
        dsl = ArchitectureDSL(f"random_arch_{random.randint(0, 1000)}")
        dsl.input([3, 32, 32])
        
        for i in range(depth):
            op = random.choice(self.available_ops)
            if op == OperationType.CONV2D:
                dsl.conv2d(
                    filters=random.choice(self.filters_options),
                    kernel_size=random.choice(self.kernel_options)
                )
            elif op == OperationType.RELU:
                dsl.relu()
            # Add more op mappings as needed
            
        dsl.linear(10)
        return dsl.build()
