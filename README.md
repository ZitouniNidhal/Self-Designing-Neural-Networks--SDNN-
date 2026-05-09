# ArchitectAI: Self-Designing Neural Networks (SDNN)

ArchitectAI is a next-generation framework for the automated discovery and optimization of neural network architectures. Unlike traditional NAS (Neural Architecture Search), SDNN leverages evolutionary algorithms combined with logic-based reasoning to "design" architectures that are optimized for specific hardware constraints and tasks.

## Key Features

- **Domain-Specific Language (DSL)**: Define search spaces using a high-level graph-based DSL.
- **Evolutionary Engine**: Multi-objective optimization for accuracy, latency, and energy efficiency.
- **Hardware-Aware**: Real-time profiling and optimization for edge devices and specialized hardware.
- **Reasoning-Driven**: Uses a logic engine to prune invalid architectures and guide the search process.
- **PyTorch Integration**: Seamlessly generates and compiles PyTorch code from discovered designs.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from architectai import Architect

# Define a search task
architect = Architect(task="image_classification", dataset="cifar10")

# Start the discovery process
best_model = architect.discover(iterations=100)

# Export to PyTorch
best_model.export("model.pt")
```

## Project Structure

- `src/architectai/core`: DSL and graph primitives.
- `src/architectai/evolution`: Evolutionary search algorithms.
- `src/architectai/reasoning`: Logic-based pruning and guidance.
- `src/architectai/codegen`: PyTorch code generation.
- `src/architectai/hardware`: Hardware profiling and constraints.
