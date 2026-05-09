import click
from rich.console import Console
from ..system.architectai import Architect
from ..system.config import config

console = Console()

@click.group()
def main():
    """ArchitectAI: Self-Designing Neural Networks CLI."""
    pass

@main.command()
@click.option("--task", default="image_classification", help="Discovery task (e.g., image_classification, nlp)")
@click.option("--iterations", default=10, help="Number of search iterations")
@click.option("--output", default="best_model.json", help="Path to save the best model")
def discover(task, iterations, output):
    """Start the architecture discovery process."""
    architect = Architect()
    graph = architect.discover(task=task, iterations=iterations)
    architect.export(graph, output)
    console.print(f"[bold green]Success![/bold green] Best architecture saved to {output}")

if __name__ == "__main__":
    main()
