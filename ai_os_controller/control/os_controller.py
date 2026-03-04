"""
control/os_controller.py

Low-level OS control layer using PyAutoGUI.

All methods simulate keyboard / mouse events.  A per-action cooldown
timer is applied to prevent accidental repeated triggering.
"""

import logging
import time

logger = logging.getLogger(__name__)

# Default cooldown between the same action (seconds)
DEFAULT_COOLDOWN = 0.8


class OSController:
    """
    Wraps PyAutoGUI calls behind a clean interface with per-action cooldowns.

    Args:
        cooldown: Minimum seconds between consecutive identical actions.
    """

    def __init__(self, cooldown: float = DEFAULT_COOLDOWN):
        self.cooldown = cooldown
        self._last_action: dict[str, float] = {}

        try:
            import pyautogui
            pyautogui.FAILSAFE = True      # move mouse to corner to abort
            pyautogui.PAUSE    = 0.02      # minimal inter-call pause
            self._pag = pyautogui
            self._available = True
        except ImportError:
            logger.error(
                "pyautogui is not installed. "
                "OS control will be disabled. Run: pip install pyautogui"
            )
            self._available = False

    # ------------------------------------------------------------------
    # Reel / media controls
    # ------------------------------------------------------------------

    def next_reel(self):
        """Scroll down to advance to the next reel / post."""
        self._act("next_reel", self._pag.scroll, -500)

    def prev_reel(self):
        """Scroll up to go back to the previous reel / post."""
        self._act("prev_reel", self._pag.scroll, 500)

    def like(self):
        """Press 'L' – Instagram / TikTok keyboard shortcut for like."""
        self._act("like", self._pag.press, "l")

    def pause(self):
        """Press Space – pause / play toggle."""
        self._act("pause", self._pag.press, "space")

    def volume_up(self):
        """Raise system volume."""
        self._act("volume_up", self._pag.press, "volumeup")

    def volume_down(self):
        """Lower system volume."""
        self._act("volume_down", self._pag.press, "volumedown")

    # ------------------------------------------------------------------
    # Cursor control
    # ------------------------------------------------------------------

    def move_cursor(self, x: float, y: float):
        """
        Move the mouse cursor to the given screen coordinates.

        Args:
            x, y: Target position in screen pixels.
        """
        if not self._available:
            return
        try:
            self._pag.moveTo(int(x), int(y), duration=0.05)
        except Exception as exc:
            logger.warning(f"move_cursor failed: {exc}")

    def click(self):
        """Perform a left mouse click."""
        self._act("click", self._pag.click)

    def right_click(self):
        """Perform a right mouse click."""
        self._act("right_click", self._pag.rightClick)

    def scroll(self, amount: int):
        """
        Scroll the mouse wheel.

        Positive amount → scroll up.
        Negative amount → scroll down.
        """
        if not self._available:
            return
        try:
            self._pag.scroll(amount)
        except Exception as exc:
            logger.warning(f"scroll failed: {exc}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _act(self, name: str, fn, *args):
        """Execute *fn* with *args* if the per-action cooldown has elapsed."""
        if not self._available:
            return

        now = time.monotonic()
        if now - self._last_action.get(name, 0.0) < self.cooldown:
            return   # still in cooldown

        self._last_action[name] = now
        logger.debug(f"OSController action: {name} {args}")
        try:
            fn(*args)
        except Exception as exc:
            logger.warning(f"OSController '{name}' failed: {exc}")
