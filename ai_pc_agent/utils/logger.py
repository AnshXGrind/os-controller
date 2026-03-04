"""ai_pc_agent/utils/logger.py

Shared logging setup for the entire ai_pc_agent package.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

try:
    from rich.logging import RichHandler
    _RICH = True
except ImportError:
    _RICH = False

_INITIALISED = False


def get_logger(name: str = "agent") -> logging.Logger:
    """Return a named logger, initialising the root config once."""
    global _INITIALISED
    if not _INITIALISED:
        _setup_root()
        _INITIALISED = True
    return logging.getLogger(name)


def _setup_root():
    root = logging.getLogger()
    if root.handlers:
        return  # already configured elsewhere
    root.setLevel(logging.DEBUG)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    if _RICH:
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    handler.setLevel(logging.INFO)
    root.addHandler(handler)

    # Optional: file handler
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "agent.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)
