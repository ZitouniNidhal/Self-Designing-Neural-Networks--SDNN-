# ArchitectAI System Architecture

ArchitectAI is structured into modular layers designed for high extensibility, search efficiency, and clean framework abstraction.

```
+-------------------------------------------------------------+
|                      Architect Class                        |
|             (Orchestrates Search & Codegen)                 |
+-------------------------------------------------------------+
       |                                             |
       v                                             v
+-----------------------+                 +-----------------------+
|  Evolution Subsystem  |                 |  Reasoning Subsystem  |
| (Mutator, Selector,   |                 |  (Rule Evaluation,    |
|   Population, Search) |                 |    Knowledge Base)    |
+-----------------------+                 +-----------------------+
       |                                             |
       +----------------------+----------------------+
                              |
                              v
              +-------------------------------+
              |    Hardware & Evaluation      |
              |  (Profiler, Device Profiles,  |
              |    Quantization, Edge Opt)    |
              +-------------------------------+
                              |
                              v
              +-------------------------------+
              |    Codegen & Export Layer     |
              |  (PyTorch Code Compiler)      |
              +-------------------------------+
```

## Core Primitives & Graph Representation
Architectures are modeled as Directed Acyclic Graphs (DAGs) using `ArchitectureGraph` backed by NetworkX. Nodes represent `Primitive` operations (`CONV2D`, `LINEAR`, `RELU`, `BATCHNORM`, `MAXPOOL2D`), and edges represent tensor flow connections.

## Evolutionary Optimization
Search spaces are sampled using `SearchSpace`. Mutations are applied via `Mutator` (kernel sizing, filter adjustment, node insertion), and recombination via `Crossover`. Selection is performed using `Selector` with support for tournament and roulette-wheel strategies.

## Reasoning Engine
Architectures are validated using `ReasoningEngine` against deterministic rules (`NoCyclesRule`, `HasInputRule`, `HasOutputRule`, `MaxDepthRule`) and pattern matching via `KnowledgeBase`.
