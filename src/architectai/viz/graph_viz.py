from typing import Optional

from ..core.graph import ArchitectureGraph


def plot_graph(
    graph: ArchitectureGraph, save_path: Optional[str] = None
) -> str:
    """Plot or summarize the architecture graph layout."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(graph.graph)
        labels = {
            n: f"{n}\n({graph.get_primitive(n).config.op})"
            for n in graph.graph.nodes()
        }
        nx.draw(
            graph.graph,
            pos,
            with_labels=True,
            labels=labels,
            node_color="skyblue",
            node_size=2000,
            arrowsize=20,
            ax=ax,
        )
        ax.set_title(f"Architecture: {graph.name}")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            return f"Graph plot saved to {save_path}"
        plt.close(fig)
        return f"Graph plot created for {graph.name}"
    except Exception:
        # Fallback text representation if GUI/matplotlib fails
        lines = [f"Graph structure for {graph.name}:"]
        for src, dst in graph.graph.edges():
            lines.append(f"  {src} -> {dst}")
        return "\n".join(lines)
