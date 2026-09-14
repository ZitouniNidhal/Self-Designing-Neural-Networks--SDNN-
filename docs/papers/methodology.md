# ArchitectAI Methodology & Theoretical Foundation

## Abstract
Self-Designing Neural Networks (SDNN) automate neural architecture search (NAS) by combining stochastic evolutionary operators with symbolic reasoning rules.

## 1. Graph Formulation
An architecture is defined as a directed acyclic graph $G = (V, E)$, where each vertex $v \in V$ represents a primitive tensor transformation $f_v$, and each edge $(u, v) \in E$ denotes tensor dependency.

## 2. Multi-Objective Fitness Evaluation
The objective function balances task accuracy and hardware constraints:

$$\text{Fitness}(G) = \alpha \cdot \text{Accuracy}(G) - \beta \cdot \max(0, \text{Params}(G) - P_{\text{max}}) - \gamma \cdot \text{Latency}(G)$$

where $P_{\text{max}}$ represents the maximum allowable parameter threshold for the target hardware device.

## 3. Symbolic Reasoning Integration
Before evaluating fitness, candidates pass through a symbolic reasoning engine. Invalid DAG topologies (e.g. cyclic connections, missing output heads) are immediately rejected, saving compute cycles.
