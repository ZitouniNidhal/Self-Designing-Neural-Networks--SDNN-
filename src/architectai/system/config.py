from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

class EvolutionConfig(BaseSettings):
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5
    
class HardwareConfig(BaseSettings):
    target_device: str = "cpu"
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None

class ArchitectConfig(BaseSettings):
    project_name: str = "SDNN_Project"
    experiment_dir: str = "experiments/results"
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    
    class Config:
        env_prefix = "ARCHITECT_"
        case_sensitive = False

# Default configuration instance
config = ArchitectConfig()
