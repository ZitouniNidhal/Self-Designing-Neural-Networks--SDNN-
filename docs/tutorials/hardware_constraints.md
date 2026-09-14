# Hardware-Constrained Optimization Tutorial

Learn how to configure parameter budgets, memory limits, and hardware optimization pipelines for edge devices.

## 1. Setting Hardware Constraints

Use `HardwareConstraints` to impose explicit physical limits:

```python
from architectai.hardware.constraints import HardwareConstraints

constraints = HardwareConstraints(
    max_params=100_000,      # Maximum 100k parameters
    max_memory_mb=64,        # Maximum 64 MB RAM footprint
    target_latency_ms=10.0   # Target 10ms latency
)
```

## 2. Applying Edge Optimization & Quantization

```python
from architectai.hardware.edge_optimizer import EdgeOptimizer
from architectai.hardware.quantization import Quantizer

# 1. Optimize graph for Jetson Nano
optimizer = EdgeOptimizer(target_device="jetson_nano")
opt_graph = optimizer.optimize(graph)

# 2. Quantize model to INT8
quantizer = Quantizer(precision="int8")
quantized_graph = quantizer.quantize(opt_graph)

# Estimate size reduction
stats = quantizer.estimate_compression(original_size_mb=4.0)
print(f"Quantized size: {stats['quantized_mb']} MB (Ratio: {stats['compression_ratio']}x)")
```
