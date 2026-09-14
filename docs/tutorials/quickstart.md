# Quickstart Tutorial

This guide shows how to discover and export an optimal neural network architecture in under 5 minutes.

## 1. Installation

```bash
pip install -e .
```

## 2. Basic Architecture Search

Create a script `quickstart.py`:

```python
from architectai import Architect

architect = Architect()

# Discover architecture for image classification
best_graph = architect.discover(
    task="image_classification",
    iterations=10
)

# Export generated PyTorch code
architect.export(best_graph, "discovered_model.py")
print("Architecture search complete! Model code generated at discovered_model.py")
```

## 3. Running the CLI

You can also discover architectures directly from your terminal:

```bash
architect discover --task image_classification --iterations 5 --output my_model.py
```
