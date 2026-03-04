"""
control/app_control.py

Launch and terminate desktop applications.

App paths are sourced from utils/command_map.APP_PATHS so all
path definitions live in one central place.

Supports:
    • Opening an app by name  (os.startfile / subprocess)
    • Closing an app by name  (taskkill /im <name>.exe /f)
    • Window focus via pygetwindow
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from jarvis_ai.utils.command_map import APP_PATHS, APP_ALIASES

logger = logging.getLogger(__name__)


class AppControl:
    """
    Opens and closes desktop applications by name.

    Args:
        extra_paths: Additional {app_name: path} mappings to merge with
                     the built-in APP_PATHS.  Useful for per-machine overrides.
    """

    def __init__(self, extra_paths: dict[str, str] = None):
        self._paths = dict(APP_PATHS)
        if extra_paths:
            self._paths.update(extra_paths)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_app(self, name: str) -> bool:
        """
        Launch an application by its registered name.

        Args:
            name: App label (e.g. 'chrome', 'vscode', 'calculator').

        Returns:
            True if the launch was attempted, False if name is unknown.
        """
        key = self._resolve(name)
        if not key:
            logger.warning(f"AppControl.open_app: unknown app '{name}'")
            return False

        path = self._paths.get(key, "") or key
        logger.info(f"AppControl: opening '{key}' → {path}")

        # Primary: shell=True works for system apps (notepad, calc, etc.)
        # without needing a full path, and works on all Windows accounts.
        try:
            subprocess.Popen(path, shell=True)
            return True
        except Exception as exc:
            logger.warning(f"AppControl: shell launch failed for '{path}': {exc}")

        # Fallback: os.startfile (Windows-only, requires a valid file path)
        if sys.platform == "win32":
            try:
                os.startfile(path)
                return True
            except FileNotFoundError:
                logger.error(
                    f"AppControl.open_app: '{path}' not found. "
                    f"Check APP_PATHS entry for '{key}'."
                )
            except OSError as exc:
                logger.error(f"AppControl.open_app os.startfile failed for '{key}': {exc}")
        else:
            logger.error(
                f"AppControl.open_app: os.startfile not available on '{sys.platform}'. "
                f"Add a native path for '{key}' in APP_PATHS."
            )
        return False

    def close_app(self, name: str) -> bool:
        """
        Force-terminate an application by name.

        Args:
            name: App label or raw process name (without .exe).

        Returns:
            True if taskkill was called, False if name is unknown.
        """
        key = self._resolve(name) or name
        exe = key if key.endswith(".exe") else f"{key}.exe"
        logger.info(f"AppControl: closing '{exe}'")
        try:
            result = subprocess.run(
                ["taskkill", "/im", exe, "/f"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"AppControl: killed '{exe}'")
                return True
            else:
                logger.warning(f"AppControl: taskkill exit {result.returncode} for '{exe}'")
                return False
        except Exception as exc:
            logger.error(f"AppControl.close_app failed: {exc}")
            return False

    def focus_app(self, name: str) -> bool:
        """
        Bring an already-open application window to the foreground.

        Args:
            name: Partial window title (case-insensitive search).

        Returns:
            True if the window was focused, False otherwise.
        """
        try:
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle(name)
            if windows:
                win = windows[0]
                win.restore()
                win.activate()
                logger.info(f"AppControl: focused window '{win.title}'")
                return True
        except ImportError:
            logger.warning("pygetwindow not installed – focus_app unavailable.")
        except Exception as exc:
            logger.warning(f"AppControl.focus_app('{name}'): {exc}")
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, name: str) -> Optional[str]:
        """
        Normalise and resolve the app name.

        1. Lower-case + strip
        2. Check aliases
        3. Check the paths dict directly
        """
        n = name.lower().strip()
        if n in APP_ALIASES:
            return APP_ALIASES[n]
        # Partial match
        for key in self._paths:
            if n == key or n in key:
                return key
        return None
