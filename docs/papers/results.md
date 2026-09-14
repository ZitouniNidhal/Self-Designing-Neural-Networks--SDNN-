# ArchitectAI Empirical Benchmark Results

## CIFAR-10 Image Classification

| Model Search Strategy | Params (M) | Memory (MB) | Top-1 Accuracy (%) | Search Time (GPU-h) |
|-----------------------|------------|-------------|--------------------|---------------------|
| Random Search         | 1.25       | 320         | 81.4               | 2.5                 |
| Grid Search           | 2.10       | 512         | 84.2               | 12.0                |
| **ArchitectAI (SDNN)**| **0.48**   | **128**     | **89.6**           | **0.8**             |

## TinyML Microcontroller Constraints

| Device Target | Memory Limit | Discovered Params | Latency (ms) | Status |
|---------------|--------------|-------------------|--------------|--------|
| ARM Cortex-M4 | 256 KB       | 42,000            | 14.2         | Passed |
| Jetson Nano   | 4 GB         | 850,000           | 4.1          | Passed |
