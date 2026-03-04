"""ai_pc_agent/control/browser_control.py

Browser automation using PyAutoGUI keyboard shortcuts.
Works with any focused browser window.
"""

from __future__ import annotations
import time
import pyautogui
import pyperclip

from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.browser")


class BrowserControl:
    """Control any focused browser window via keyboard shortcuts."""

    def __init__(self, scroll_amount: int = 300):
        self.scroll_amount = scroll_amount

    # ── Tabs ──────────────────────────────────────────────────────────────────

    def new_tab(self) -> bool:
        return self._hotkey("ctrl", "t", label="new tab")

    def close_tab(self) -> bool:
        return self._hotkey("ctrl", "w", label="close tab")

    def next_tab(self) -> bool:
        return self._hotkey("ctrl", "tab", label="next tab")

    def prev_tab(self) -> bool:
        return self._hotkey("ctrl", "shift", "tab", label="prev tab")

    def reopen_tab(self) -> bool:
        return self._hotkey("ctrl", "shift", "t", label="reopen tab")

    # ── Navigation ────────────────────────────────────────────────────────────

    def go_back(self) -> bool:
        return self._hotkey("alt", "left", label="back")

    def go_forward(self) -> bool:
        return self._hotkey("alt", "right", label="forward")

    def refresh(self) -> bool:
        return self._hotkey("f5", label="refresh")

    def stop_loading(self) -> bool:
        return self._hotkey("escape", label="stop loading")

    def open_url(self, url: str) -> bool:
        try:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            pyautogui.hotkey("ctrl", "l")
            time.sleep(0.3)
            pyperclip.copy(url)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.2)
            pyautogui.press("enter")
            logger.info("Opened URL: %s", url)
            return True
        except Exception as e:
            logger.error("open_url: %s", e); return False

    # ── Search ────────────────────────────────────────────────────────────────

    def search_google(self, query: str) -> bool:
        return self.open_url(f"https://www.google.com/search?q={query.replace(' ', '+')}")

    def search_youtube(self, query: str) -> bool:
        return self.open_url(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")

    def search_github(self, query: str) -> bool:
        return self.open_url(f"https://github.com/search?q={query.replace(' ', '+')}")

    # ── Scroll ────────────────────────────────────────────────────────────────

    def scroll_up(self, amount: int | None = None) -> bool:
        try:
            pyautogui.scroll(amount or self.scroll_amount)
            return True
        except Exception as e:
            logger.error("scroll_up: %s", e); return False

    def scroll_down(self, amount: int | None = None) -> bool:
        try:
            pyautogui.scroll(-(amount or self.scroll_amount))
            return True
        except Exception as e:
            logger.error("scroll_down: %s", e); return False

    # ── Window management ─────────────────────────────────────────────────────

    def minimize_window(self) -> bool:
        return self._hotkey("win", "down", label="minimize")

    def maximize_window(self) -> bool:
        return self._hotkey("win", "up", label="maximize")

    def close_window(self) -> bool:
        return self._hotkey("alt", "f4", label="close window")

    def switch_window(self) -> bool:
        return self._hotkey("alt", "tab", label="switch window")

    def fullscreen(self) -> bool:
        return self._hotkey("f11", label="fullscreen")

    # ── Dev tools ─────────────────────────────────────────────────────────────

    def open_devtools(self) -> bool:
        return self._hotkey("f12", label="devtools")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _hotkey(self, *keys: str, label: str = "") -> bool:
        try:
            pyautogui.hotkey(*keys)
            logger.info("Browser: %s", label or "+".join(keys))
            return True
        except Exception as e:
            logger.error("hotkey %s: %s", keys, e)
            return False
