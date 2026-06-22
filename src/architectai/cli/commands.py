import click
from rich.console import Console
from .args import CLIArgs
from ..system.architectai import Architect

console = Console()

@click.group()
def cli():
    """ArchitectAI command group."""
    pass

@cli.command()
@click.option("--task", default=CLIArgs.task, help="Discovery task (e.g. image_classification, nlp)")
@click.option("--iterations", default=CLIArgs.iterations, type=int, help="Number of discovery iterations")
@click.option("--output", default=CLIArgs.output, help="Path to save the exported architecture")
def discover(task: str, iterations: int, output: str):
    """Run architecture discovery and save the best design."""
    architect = Architect()
    graph = architect.discover(task=task, iterations=iterations)
    architect.export(graph, output)
    console.print(f"[bold green]Success![/bold green] Best architecture saved to {output}")
