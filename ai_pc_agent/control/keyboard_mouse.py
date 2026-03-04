"""ai_pc_agent/control/keyboard_mouse.py

Low-level keyboard and mouse automation via PyAutoGUI.
"""

from __future__ import annotations
import time
import pyautogui
import pyperclip

from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.input")

# Safety: prevent pyautogui from crashing on edge of screen
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.05


class KeyboardMouse:

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def press_key(self, key: str) -> bool:
        try:
            pyautogui.press(key)
            logger.info("Key press: %s", key)
            return True
        except Exception as e:
            logger.error("press_key '%s': %s", key, e); return False

    def hotkey(self, *keys: str) -> bool:
        try:
            pyautogui.hotkey(*keys)
            logger.info("Hotkey: %s", "+".join(keys))
            return True
        except Exception as e:
            logger.error("hotkey %s: %s", keys, e); return False

    def type_text(self, text: str, interval: float = 0.03) -> bool:
        try:
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            logger.info("Typed text (len=%d)", len(text))
            return True
        except Exception as e:
            logger.error("type_text: %s", e); return False

    def type_slowly(self, text: str, interval: float = 0.05) -> bool:
        try:
            pyautogui.typewrite(text, interval=interval)
            return True
        except Exception as e:
            logger.error("type_slowly: %s", e); return False

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def move_mouse(self, x: int, y: int, duration: float = 0.3) -> bool:
        try:
            pyautogui.moveTo(x, y, duration=duration)
            logger.info("Mouse → (%d, %d)", x, y)
            return True
        except Exception as e:
            logger.error("move_mouse: %s", e); return False

    def click_mouse(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
    ) -> bool:
        try:
            if x is not None and y is not None:
                pyautogui.click(x, y, button=button, clicks=clicks)
            else:
                pyautogui.click(button=button, clicks=clicks)
            logger.info("Click %s (%s, %s)", button, x, y)
            return True
        except Exception as e:
            logger.error("click_mouse: %s", e); return False

    def right_click(self, x: int | None = None, y: int | None = None) -> bool:
        return self.click_mouse(x, y, button="right")

    def double_click(self, x: int | None = None, y: int | None = None) -> bool:
        return self.click_mouse(x, y, clicks=2)

    def scroll(self, amount: int) -> bool:
        """Positive = scroll up, negative = scroll down."""
        try:
            pyautogui.scroll(amount)
            return True
        except Exception as e:
            logger.error("scroll: %s", e); return False

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> bool:
        try:
            pyautogui.drag(x2 - x1, y2 - y1, duration=duration, button="left")
            return True
        except Exception as e:
            logger.error("drag: %s", e); return False

    def get_position(self) -> tuple[int, int]:
        pos = pyautogui.position()
        return pos.x, pos.y

    def get_screen_size(self) -> tuple[int, int]:
        size = pyautogui.size()
        return size.width, size.height
