

# 🧠 ArchitectAI

### Self-Designing Neural Networks (SDNN)

[![Python Tests](https://github.com/ZitouniNidhal/Self-Designing-Neural-Networks--SDNN-/actions/workflows/tests.yml/badge.svg)](https://github.com/ZitouniNidhal/Self-Designing-Neural-Networks--SDNN-/actions/workflows/tests.yml)
[![Lint](https://github.com/ZitouniNidhal/Self-Designing-Neural-Networks--SDNN-/actions/workflows/lint.yml/badge.svg)](https://github.com/ZitouniNidhal/Self-Designing-Neural-Networks--SDNN-/actions/workflows/lint.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A next-generation framework for automated discovery and optimization of neural network architectures.**

Unlike traditional NAS (Neural Architecture Search), SDNN combines evolutionary algorithms with logic-based reasoning to *design* architectures optimized for specific hardware constraints and tasks.

[Getting Started](#-getting-started) · [Examples](#-examples) · [Documentation](#-documentation) · [Contributing](#-contributing)

</div>

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Graph-Based DSL** | Define and compose neural architectures using a fluent, chainable API built on DAG primitives |
| **Evolutionary Search** | Multi-objective optimization balancing accuracy, latency, and energy efficiency |
| **Hardware-Aware** | Constrain search by parameter count, memory budget, and target device (CPU/GPU/edge) |
| **Reasoning Engine** | Logic-based pruning to eliminate invalid architectures and guide the search process |
| **PyTorch Code Generation** | Automatically compile discovered architectures into runnable PyTorch modules |
| **REST API** | Serve architecture discovery as a service via FastAPI |
| **CLI** | Full command-line interface for running discovery pipelines |
| **Visualization** | Plot evolution progress, architecture graphs, and Pareto fronts |

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     ArchitectAI                          │
├──────────┬───────────┬────────────┬───────────┬──────────┤
│   Core   │ Evolution │ Reasoning  │  CodeGen  │ Hardware │
│  ──────  │  ───────  │  ────────  │  ──────── │ ──────── │
│  DSL     │ Search    │ Engine     │ Compiler  │ Profiler │
│  Graph   │ Space     │ Rules      │ PyTorch   │ Constr.  │
│  Prims   │ Populat.  │ Knowl.Base │ Optimizer │ Evaluator│
├──────────┴───────────┴────────────┴───────────┴──────────┤
│              CLI  ·  REST API  ·  Visualization          │
└──────────────────────────────────────────────────────────┘
```

The framework is built around a **DAG-based architecture graph** where nodes represent neural network operations (`Conv2D`, `ReLU`, `Linear`, `BatchNorm`, etc.) and edges represent data flow. The evolutionary engine samples candidate architectures from a configurable search space, evaluates them against hardware constraints, and iteratively refines the population toward optimal designs.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- PyTorch 2.0+

### Installation

```bash
# Clone the repository
git clone https://github.com/ZitouniNidhal/Self-Designing-Neural-Networks--SDNN-.git
cd Self-Designing-Neural-Networks--SDNN-

# Install in editable mode
pip install -e .

# Install dev dependencies (for testing & linting)
pip install -r requirements-dev.txt
```

### Quick Start

```python
from architectai import Architect
from architectai.hardware.constraints import HardwareConstraints

# Initialize the architect
architect = Architect()

# Define hardware constraints
constraints = HardwareConstraints(
    max_params=500_000,
    max_memory_mb=256,
    target_device="cpu",
)

# Run architecture discovery
best_architecture = architect.discover(
    task="image_classification",
    iterations=50,
    constraints=constraints,
)

# Export to PyTorch code
architect.export(best_architecture, "my_model.py")
```

---

## 🔧 Usage

### Python API

#### Building Architectures with the DSL

The fluent DSL API lets you define architectures as chained operations:

```python
from architectai.core.dsl import ArchitectureDSL

# Build a simple CNN
model = (
    ArchitectureDSL("my_cnn")
    .input([3, 32, 32])
    .conv2d(32, kernel_size=3, stride=1)
    .relu()
    .conv2d(64)
    .relu()
    .linear(10)
    .build()
)

# Inspect the architecture graph
print(model.to_dict())
```

#### Exploring the Search Space

```python
from architectai.evolution.search_space import SearchSpace

space = SearchSpace()
# Sample a random architecture with depth 6
candidate = space.sample_random_architecture(depth=6)
```

### CLI

The `architect` CLI is installed as a console script:

```bash
# Run architecture discovery
architect discover --task image_classification --iterations 100 --output model.py

# Show help
architect --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | `image_classification` | Discovery task type |
| `--iterations` | `10` | Number of search iterations |
| `--output` | `output_architecture.py` | Path to save the exported architecture |

### REST API

Start the API server to expose discovery as a service:

```bash
uvicorn architectai.api.server:app --reload
```

### Docker

```bash
cd docker
docker-compose up --build
```

---

## 📂 Project Structure

```
Self-Designing-Neural-Networks--SDNN-/
├── src/architectai/
│   ├── __init__.py              # Package exports
│   ├── core/                    # Foundation layer
│   │   ├── primitives.py        #   Operation types, NodeConfig, Primitive
│   │   ├── graph.py             #   ArchitectureGraph (NetworkX DAG)
│   │   └── dsl.py               #   Fluent DSL for building architectures
│   ├── evolution/               # Evolutionary search
│   │   ├── search_space.py      #   Defines available ops & sampling
│   │   └── population.py        #   Population management
│   ├── reasoning/               # Logic-based guidance
│   │   ├── engine.py            #   Reasoning engine
│   │   ├── rules.py             #   Architecture validity rules
│   │   ├── knowledge_base.py    #   Domain knowledge store
│   │   └── explainer.py         #   Decision explanations
│   ├── codegen/                 # Code generation
│   │   ├── compiler.py          #   Graph → file export (JSON / Python)
│   │   ├── pytorch_generator.py #   Graph → PyTorch nn.Module source
│   │   └── optimizer.py         #   Graph-level optimizations
│   ├── hardware/                # Hardware awareness
│   │   ├── constraints.py       #   HardwareConstraints & evaluator
│   │   └── profiler.py          #   Runtime profiling
│   ├── evaluation/              # Model evaluation
│   ├── viz/                     # Visualization utilities
│   │   ├── graph_viz.py         #   Architecture graph plots
│   │   ├── evolution_viz.py     #   Evolution progress charts
│   │   ├── pareto_viz.py        #   Multi-objective Pareto fronts
│   │   └── report_generator.py  #   HTML/PDF report generation
│   ├── cli/                     # Command-line interface
│   │   ├── main.py              #   Click CLI entry point
│   │   ├── commands.py          #   CLI commands (discover)
│   │   └── args.py              #   Default arguments
│   ├── api/                     # REST API
│   │   ├── server.py            #   FastAPI application
│   │   ├── endpoints.py         #   API routes
│   │   └── schemas.py           #   Request/response models
│   ├── system/                  # System coordination
│   │   ├── architectai.py       #   Architect (main entry class)
│   │   ├── config.py            #   ArchitectConfig (pydantic-settings)
│   │   └── logger.py            #   Rich-powered logging
│   └── utils/                   # Shared utilities
├── examples/
│   ├── image_classification/    # CIFAR-10, ImageNet examples
│   ├── nlp/                     # NLP task examples
│   ├── multimodal/              # Vision-language examples
│   └── edge/                    # Edge deployment examples
├── tests/                       # Pytest test suite
├── docs/                        # Documentation
├── docker/                      # Docker & Compose config
├── experiments/                 # Experiment outputs
├── models/                      # Saved models
├── .github/workflows/           # CI: lint, test, publish
├── pyproject.toml               # Build config & tool settings
├── requirements.txt             # Runtime dependencies
└── requirements-dev.txt         # Dev/test dependencies
```

---

## 📚 Examples

Ready-to-run examples are provided for multiple domains:

| Example | Path | Description |
|---------|------|-------------|
| CIFAR-10 Classification | [`examples/image_classification/discover_cifar10.py`](examples/image_classification/discover_cifar10.py) | Discover a CNN for CIFAR-10 with parameter & memory constraints |
| ImageNet Discovery | [`examples/image_classification/discover_imagenet.py`](examples/image_classification/discover_imagenet.py) | Large-scale image classification architecture search |
| Vision-Language | [`examples/multimodal/discover_vision_language.py`](examples/multimodal/discover_vision_language.py) | Multimodal architecture discovery |
| Edge Deployment | [`examples/edge/`](examples/edge/) | Hardware-constrained search for edge devices |

```bash
# Run the CIFAR-10 example
python examples/image_classification/discover_cifar10.py
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=architectai

# Run a specific test file
pytest tests/test_graph.py -v
```

### Linting & Type Checking

```bash
# Linting
flake8 src tests

# Type checking
mypy src

# Formatting
black src tests
isort src tests
```

---

## 🛠️ Configuration

ArchitectAI uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration. You can configure the framework via environment variables or by passing an `ArchitectConfig` object:

```python
from architectai import Architect, ArchitectConfig

config = ArchitectConfig(
    project_name="my_experiment",
    debug=True,
)
architect = Architect(config_override=config)
```

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ARCHITECTAI_PROJECT_NAME` | `sdnn_project` | Name of the project/experiment |
| `ARCHITECTAI_DEBUG` | `false` | Enable debug logging |

---

## 🗺️ Roadmap

- [x] Core DSL and graph representation
- [x] Evolutionary search with random sampling
- [x] Hardware-aware constraint evaluation
- [x] PyTorch code generation
- [x] CLI and REST API
- [x] CI/CD pipeline (lint, test, publish)
- [ ] NSGA-II multi-objective evolution
- [ ] Full reasoning engine with architecture validity rules
- [ ] Training loop integration with real accuracy evaluation
- [ ] Latency and memory profiling on real hardware
- [ ] Visualization dashboards (graph, evolution, Pareto)
- [ ] ONNX and TensorFlow export backends
- [ ] Distributed search across multiple GPUs/nodes

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Ensure all checks pass:
   ```bash
   flake8 src tests && mypy src && pytest
   ```
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [ZitouniNidhal](https://github.com/ZitouniNidhal)


