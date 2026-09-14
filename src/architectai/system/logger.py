import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(name: str = "architectai", level: str = "INFO"):
    """Initialize a beautiful logger using Rich."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        console = Console()
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logger()
