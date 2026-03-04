"""ai_pc_agent/diagnostics/error_handler.py

Global exception interception, structured error reporting, and
optional self-healing via the coding LLM.
"""

from __future__ import annotations
import sys
import traceback
import threading
from typing import Callable

from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.errors")


class ErrorHandler:
    """
    Centralised error handler.
    - Installs a global sys.excepthook
    - Provides a decorator/context manager for guarded execution
    - Optionally forwards error info to a self-healing callback
    """

    def __init__(self, heal_callback: Callable[[str, str], str] | None = None):
        """
        heal_callback(code, traceback_str) → suggested_fix_str
        Set to None to disable auto-healing.
        """
        self._heal_cb  = heal_callback
        self._handlers: list[Callable] = []
        self._install_global_hook()

    # ── Global hook ───────────────────────────────────────────────────────────

    def _install_global_hook(self):
        def _hook(exc_type, exc_value, exc_tb):
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            logger.critical("UNHANDLED EXCEPTION:\n%s", tb)
            for h in self._handlers:
                try:
                    h(exc_type, exc_value, exc_tb)
                except Exception:
                    pass
        sys.excepthook = _hook

        def _thread_hook(args):
            tb = "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback
            ))
            logger.critical("UNHANDLED THREAD EXCEPTION:\n%s", tb)
        threading.excepthook = _thread_hook

    def add_handler(self, fn: Callable):
        self._handlers.append(fn)

    # ── Guarded execution ─────────────────────────────────────────────────────

    def safe_call(
        self,
        fn: Callable,
        *args,
        default=None,
        label: str = "",
        **kwargs,
    ):
        """Call *fn* safely; return *default* on exception."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[%s] %s\n%s", label or fn.__name__, exc, tb)
            if self._heal_cb:
                try:
                    import inspect
                    code = inspect.getsource(fn)
                    suggestion = self._heal_cb(code, tb)
                    if suggestion:
                        logger.info("[%s] Heal suggestion:\n%s", label or fn.__name__, suggestion[:400])
                except Exception:
                    pass
            return default

    def guarded(self, label: str = "", default=None):
        """Decorator factory for safe execution."""
        def decorator(fn):
            def wrapper(*args, **kwargs):
                return self.safe_call(fn, *args, default=default, label=label, **kwargs)
            wrapper.__name__ = fn.__name__
            return wrapper
        return decorator

    # ── Error formatting ──────────────────────────────────────────────────────

    @staticmethod
    def format_error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"

    @staticmethod
    def last_traceback() -> str:
        return traceback.format_exc()
