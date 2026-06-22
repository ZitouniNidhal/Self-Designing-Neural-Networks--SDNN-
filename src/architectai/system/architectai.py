import random
from typing import Optional, List
from .logger import logger
from .config import config, ArchitectConfig
from ..core.graph import ArchitectureGraph
from ..core.dsl import ArchitectureDSL
from ..evolution.search_space import SearchSpace
from ..hardware.constraints import HardwareConstraints, HardwareEvaluator

class Architect:
    """
    The main coordinator for the Self-Designing Neural Networks (SDNN) framework.
    """
    
    def __init__(self, config_override: Optional[ArchitectConfig] = None):
        self.config = config_override or config
        logger.info(f"🚀 Initializing [bold blue]ArchitectAI[/bold blue] - Project: [green]{self.config.project_name}[/green]")
        self.search_space = SearchSpace()
        self.best_graph: Optional[ArchitectureGraph] = None

    def discover(self, 
                 task: str, 
                 iterations: int = 10, 
                 constraints: Optional[HardwareConstraints] = None) -> ArchitectureGraph:
        """
        Start the automated architecture discovery process.
        """
        logger.info(f"🔍 Starting discovery for task: [yellow]{task}[/yellow] with [cyan]{iterations}[/cyan] iterations")
        
        evaluator = HardwareEvaluator(constraints or HardwareConstraints())
        best_score = -1
        
        for i in range(iterations):
            # Sample a candidate
            candidate = self.search_space.sample_random_architecture(depth=random.randint(3, 8))
            
            # Check constraints
            if evaluator.satisfies(candidate):
                # In a real scenario, we would train and evaluate accuracy here.
                # For this demo, we'll use a random score if it fits the hardware.
                score = random.random()
                if score > best_score:
                    best_score = score
                    self.best_graph = candidate
                    logger.debug(f"Iter {i}: Found new best architecture (score: {score:.4f})")
            else:
                logger.debug(f"Iter {i}: Candidate rejected by hardware constraints")
        
        if self.best_graph:
            logger.info(f"✨ Discovery complete. Best architecture found with score: [green]{best_score:.4f}[/green]")
        else:
            logger.warning("⚠️ No architecture found that satisfies the given constraints.")
            
        return self.best_graph

    def export(self, graph: ArchitectureGraph, path: str):
        """
        Export the architecture graph to a file or compile to code.
        """
        if graph is None:
            raise ValueError("No architecture graph available to export.")

        logger.info(f"📦 Exporting best architecture to [magenta]{path}[/magenta]")
        from ..codegen.compiler import Compiler

        compiler = Compiler()
        compiler.compile_to_file(graph, path)
