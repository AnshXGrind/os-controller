"""
control/browser_control.py

Browser interaction via PyAutoGUI keyboard shortcuts.

Works with any Chromium-based browser (Chrome, Edge, Brave) as well
as Firefox, since all share the same standard keyboard shortcuts.

Supported actions:
    new_tab()        Ctrl+T
    close_tab()      Ctrl+W
    next_tab()       Ctrl+Tab
    previous_tab()   Ctrl+Shift+Tab
    go_back()        Alt+Left
    go_forward()     Alt+Right
    refresh()        F5
    scroll_up()      Page Up / Up arrow
    scroll_down()    Page Down / Down arrow
    open_url(url)    Focus address bar and type URL
    search(query)    Open new tab and navigate to Google search
    zoom_in()        Ctrl++
    zoom_out()       Ctrl+-
    fullscreen()     F11
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

GOOGLE_SEARCH_URL = "https://www.google.com/search?q="


class BrowserControl:
    """
    Controls the foreground browser window using keyboard shortcuts.

    Args:
        scroll_amount: Pixels to scroll per call (PyAutoGUI scroll units).
    """

    def __init__(self, scroll_amount: int = 400):
        self.scroll_amount = scroll_amount
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE    = 0.05
            self._pag    = pyautogui
            self._pag_ok = True
        except ImportError:
            logger.error("pyautogui not installed – browser control disabled.")
            self._pag_ok = False

    # ------------------------------------------------------------------
    # Tab management
    # ------------------------------------------------------------------

    def new_tab(self):
        """Open a new browser tab (Ctrl+T)."""
        self._hotkey("ctrl", "t")

    def close_tab(self):
        """Close the current tab (Ctrl+W)."""
        self._hotkey("ctrl", "w")

    def next_tab(self):
        """Switch to the next tab (Ctrl+Tab)."""
        self._hotkey("ctrl", "tab")

    def previous_tab(self):
        """Switch to the previous tab (Ctrl+Shift+Tab)."""
        self._hotkey("ctrl", "shift", "tab")

    def reopen_tab(self):
        """Reopen last closed tab (Ctrl+Shift+T)."""
        self._hotkey("ctrl", "shift", "t")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def go_back(self):
        """Navigate back in browser history (Alt+Left)."""
        self._hotkey("alt", "left")

    def go_forward(self):
        """Navigate forward in browser history (Alt+Right)."""
        self._hotkey("alt", "right")

    def refresh(self):
        """Refresh the current page (F5)."""
        self._key("f5")

    def open_url(self, url: str):
        """
        Navigate the browser to a specific URL.

        Opens the address bar with Ctrl+L, types the URL, then presses Enter.
        """
        if not self._pag_ok:
            return
        logger.info(f"BrowserControl: navigating to '{url}'")
        self._hotkey("ctrl", "l")
        time.sleep(0.3)
        self._pag.typewrite(url, interval=0.03)
        self._pag.press("enter")

    def search(self, query: str):
        """
        Open a new tab and perform a Google search.

        Args:
            query: The search string (spaces are converted to '+').
        """
        encoded = query.replace(" ", "+")
        url     = GOOGLE_SEARCH_URL + encoded
        self.new_tab()
        time.sleep(0.4)
        self.open_url(url)
        logger.info(f"BrowserControl: searching for '{query}'")

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    def scroll_down(self):
        """Scroll the page down."""
        if self._pag_ok:
            self._pag.scroll(-self.scroll_amount)

    def scroll_up(self):
        """Scroll the page up."""
        if self._pag_ok:
            self._pag.scroll(self.scroll_amount)

    # ------------------------------------------------------------------
    # View
    # ------------------------------------------------------------------

    def zoom_in(self):
        """Zoom in (Ctrl++)."""
        self._hotkey("ctrl", "+")

    def zoom_out(self):
        """Zoom out (Ctrl+-)."""
        self._hotkey("ctrl", "-")

    def reset_zoom(self):
        """Reset zoom to 100% (Ctrl+0)."""
        self._hotkey("ctrl", "0")

    def fullscreen(self):
        """Toggle full-screen mode (F11)."""
        self._key("f11")

    # ------------------------------------------------------------------
    # Window management helpers
    # ------------------------------------------------------------------

    def minimize_window(self):
        """Minimise the active window (Win+D is not ideal; use hotkey)."""
        self._hotkey("win", "down")

    def maximize_window(self):
        """Maximise the active window."""
        self._hotkey("win", "up")

    def close_window(self):
        """Close the active window (Alt+F4)."""
        self._hotkey("alt", "f4")

    def switch_window(self):
        """Alt-Tab to the next open window."""
        self._hotkey("alt", "tab")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hotkey(self, *keys):
        if self._pag_ok:
            try:
                self._pag.hotkey(*keys)
            except Exception as exc:
                logger.warning(f"BrowserControl hotkey {keys}: {exc}")

    def _key(self, key: str):
        if self._pag_ok:
            try:
                self._pag.press(key)
            except Exception as exc:
                logger.warning(f"BrowserControl press '{key}': {exc}")
