# ArchitectAI API Reference

## High-Level Orchestration

### `Architect`
Main entry point for discovery and export.

```python
from architectai import Architect

architect = Architect()
graph = architect.discover(task="image_classification", iterations=10)
architect.export(graph, "output_model.py", framework="pytorch")
```

## Core & DSL

### `ArchitectureDSL`
Fluent builder API for designing neural networks.

```python
from architectai.core.dsl import ArchitectureDSL

dsl = ArchitectureDSL("my_model")
graph = dsl.input([3, 32, 32]).conv2d(32).relu().linear(10).build()
```

## Hardware & Constraints

### `HardwareConstraints`
Specifies memory, parameter, and latency bounds.

```python
from architectai.hardware.constraints import HardwareConstraints

constraints = HardwareConstraints(
    max_params=500_000,
    max_memory_mb=256,
    target_latency_ms=15.0
)
```

## REST API Endpoints

- `POST /api/v1/discover`: Trigger architecture discovery session
- `POST /api/v1/export`: Compile architecture graph to framework code
- `GET /health`: Health status endpoint
