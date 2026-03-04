"""ai_pc_agent/diagnostics/logger.py

Diagnostics-layer logger — structured, coloured, rate-limited warnings.
Re-exports get_logger from utils.logger for convenience.
"""

from ai_pc_agent.utils.logger import get_logger

__all__ = ["get_logger"]
