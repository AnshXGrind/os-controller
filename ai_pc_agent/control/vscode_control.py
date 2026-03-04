"""ai_pc_agent/control/vscode_control.py

Automate VS Code via keyboard shortcuts and the command palette.
"""

from __future__ import annotations
import subprocess
import time
from pathlib import Path

import pyautogui
import pyperclip

from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.vscode")


class VSCodeControl:
    """Control VS Code using keyboard shortcuts and the command palette."""

    def _palette(self, command: str, delay: float = 0.4) -> bool:
        """Open command palette and run a VS Code command."""
        try:
            pyautogui.hotkey("ctrl", "shift", "p")
            time.sleep(delay)
            pyperclip.copy(command)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.3)
            pyautogui.press("enter")
            logger.info("VS Code palette: %s", command)
            return True
        except Exception as e:
            logger.error("palette '%s': %s", command, e)
            return False

    # ── File operations ───────────────────────────────────────────────────────

    def open_file(self, path: str) -> bool:
        try:
            subprocess.Popen(["code", path], shell=True)
            logger.info("VS Code opened: %s", path)
            return True
        except Exception as e:
            logger.error("open_file '%s': %s", path, e)
            return False

    def open_folder(self, path: str) -> bool:
        try:
            p = Path(path)
            subprocess.Popen(["code", str(p)], shell=True)
            return True
        except Exception as e:
            logger.error("open_folder: %s", e); return False

    def new_file(self) -> bool:
        return self._hotkey("ctrl", "n", label="new file")

    def save_file(self) -> bool:
        return self._hotkey("ctrl", "s", label="save")

    def save_all(self) -> bool:
        return self._hotkey("ctrl", "k", "s", label="save all")

    # ── Editor ────────────────────────────────────────────────────────────────

    def write_code(self, code: str) -> bool:
        """Paste code into the currently focused editor."""
        try:
            pyperclip.copy(code)
            pyautogui.hotkey("ctrl", "v")
            logger.info("Code pasted (%d chars)", len(code))
            return True
        except Exception as e:
            logger.error("write_code: %s", e); return False

    def select_all(self) -> bool:
        return self._hotkey("ctrl", "a", label="select all")

    def format_document(self) -> bool:
        return self._hotkey("shift", "alt", "f", label="format")

    def toggle_comment(self) -> bool:
        return self._hotkey("ctrl", "/", label="toggle comment")

    def undo(self) -> bool:
        return self._hotkey("ctrl", "z", label="undo")

    def redo(self) -> bool:
        return self._hotkey("ctrl", "y", label="redo")

    # ── Terminal ──────────────────────────────────────────────────────────────

    def open_terminal(self) -> bool:
        return self._hotkey("ctrl", "`", label="terminal")

    def run_terminal_command(self, cmd: str) -> bool:
        try:
            self.open_terminal()
            time.sleep(0.5)
            pyperclip.copy(cmd)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.2)
            pyautogui.press("enter")
            logger.info("Terminal command: %s", cmd[:80])
            return True
        except Exception as e:
            logger.error("run_terminal_command: %s", e); return False

    # ── Run / Debug ───────────────────────────────────────────────────────────

    def run_code(self) -> bool:
        """Run current file (requires Code Runner extension or run shortcut)."""
        return self._hotkey("ctrl", "alt", "n", label="run code")

    def start_debug(self) -> bool:
        return self._hotkey("f5", label="start debug")

    def stop_debug(self) -> bool:
        return self._hotkey("shift", "f5", label="stop debug")

    def step_over(self) -> bool:
        return self._hotkey("f10", label="step over")

    def step_into(self) -> bool:
        return self._hotkey("f11", label="step into")

    # ── Navigation ────────────────────────────────────────────────────────────

    def go_to_line(self, line: int) -> bool:
        try:
            pyautogui.hotkey("ctrl", "g")
            time.sleep(0.3)
            pyautogui.typewrite(str(line), interval=0.05)
            pyautogui.press("enter")
            return True
        except Exception as e:
            logger.error("go_to_line: %s", e); return False

    def quick_open(self, filename: str) -> bool:
        try:
            pyautogui.hotkey("ctrl", "p")
            time.sleep(0.3)
            pyperclip.copy(filename)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.3)
            pyautogui.press("enter")
            return True
        except Exception as e:
            logger.error("quick_open: %s", e); return False

    def split_editor(self) -> bool:
        return self._hotkey("ctrl", "\\", label="split editor")

    def close_editor(self) -> bool:
        return self._hotkey("ctrl", "w", label="close editor")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _hotkey(self, *keys: str, label: str = "") -> bool:
        try:
            pyautogui.hotkey(*keys)
            logger.info("VS Code: %s", label or "+".join(keys))
            return True
        except Exception as e:
            logger.error("hotkey %s: %s", keys, e); return False
