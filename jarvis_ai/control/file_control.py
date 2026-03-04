"""
control/file_control.py

File and folder management for voice-triggered actions.

Capabilities:
    • Open well-known folders (Documents, Downloads, Desktop, …)
    • Open Windows search to find a file by name
    • Create a new blank text file on the Desktop
    • Open a specific file path

All folder shortcuts come from utils/command_map.FOLDER_PATHS so they
adapt to the current user's profile directory automatically.
"""

import logging
import os
import subprocess
import time
from pathlib import Path

from jarvis_ai.utils.command_map import FOLDER_PATHS

logger = logging.getLogger(__name__)


class FileControl:
    """
    Voice-driven file and folder helper.

    Args:
        default_folder: Base folder for create_file when no path is given.
                        Defaults to the user's Desktop.
    """

    def __init__(self, default_folder: str = None):
        self.default_folder = default_folder or FOLDER_PATHS.get(
            "desktop", str(Path.home() / "Desktop")
        )

    # ------------------------------------------------------------------
    # Folder navigation
    # ------------------------------------------------------------------

    def open_folder(self, name: str = "documents") -> bool:
        """
        Open a well-known folder in Windows Explorer.

        Args:
            name: Folder alias matching a key in FOLDER_PATHS
                  ('desktop', 'documents', 'downloads', 'pictures',
                   'music', 'videos').

        Returns:
            True if Explorer was launched successfully.
        """
        key  = name.lower().strip()
        path = FOLDER_PATHS.get(key)

        if not path:
            # Try interpreting `name` as a literal path
            if Path(name).exists():
                path = name
            else:
                logger.warning(f"FileControl: unknown folder '{name}'")
                return False

        logger.info(f"FileControl: opening folder '{path}'")
        try:
            os.startfile(path)
            return True
        except Exception as exc:
            logger.error(f"FileControl.open_folder failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # File search
    # ------------------------------------------------------------------

    def search_file(self, query: str = "") -> bool:
        """
        Open Windows Search (Win+S) and type the query.

        Args:
            query: The search term to type.

        Returns:
            True if the search was triggered.
        """
        logger.info(f"FileControl: searching for '{query}'")
        try:
            import pyautogui
            pyautogui.hotkey("win", "s")
            time.sleep(0.5)
            if query:
                pyautogui.typewrite(query, interval=0.05)
            return True
        except ImportError:
            logger.error("pyautogui not installed – search_file disabled.")
            return False
        except Exception as exc:
            logger.error(f"FileControl.search_file failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # File creation
    # ------------------------------------------------------------------

    def create_file(
        self,
        filename: str  = None,
        folder:   str  = None,
        open_after: bool = True,
    ) -> str:
        """
        Create a blank text file and optionally open it.

        Args:
            filename:   Name for the new file (defaults to timestamped name).
            folder:     Directory to create it in (defaults to Desktop).
            open_after: If True, open the file in the default text editor.

        Returns:
            Absolute path of the created file, or "" on failure.
        """
        import time as _time
        if not filename:
            ts       = _time.strftime("%Y%m%d_%H%M%S")
            filename = f"jarvis_note_{ts}.txt"

        target_dir = Path(folder or self.default_folder)
        target_dir.mkdir(parents=True, exist_ok=True)

        file_path = target_dir / filename
        try:
            file_path.touch(exist_ok=True)
            logger.info(f"FileControl: created '{file_path}'")
            if open_after:
                os.startfile(str(file_path))
            return str(file_path)
        except Exception as exc:
            logger.error(f"FileControl.create_file failed: {exc}")
            return ""

    # ------------------------------------------------------------------
    # Open specific file
    # ------------------------------------------------------------------

    def open_file(self, path: str) -> bool:
        """
        Open a file at the given path using the default application.

        Args:
            path: Absolute or relative file path.

        Returns:
            True on success.
        """
        try:
            os.startfile(path)
            logger.info(f"FileControl: opened '{path}'")
            return True
        except Exception as exc:
            logger.error(f"FileControl.open_file('{path}'): {exc}")
            return False
