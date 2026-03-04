"""ai_pc_agent/vision/screen_capture.py

Screen capture using mss (primary) with pyautogui fallback.
"""

from __future__ import annotations
import time
from pathlib import Path

from ai_pc_agent.utils import config
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.vision")


class ScreenCapture:
    """Capture full-screen or region screenshots."""

    def __init__(self, save_dir: str | None = None):
        self.save_dir = Path(save_dir or config.get("SCREENSHOT_DIR", "screenshots"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        try:
            import mss
            return "mss"
        except ImportError:
            pass
        try:
            import pyautogui
            return "pyautogui"
        except ImportError:
            pass
        logger.error("Neither mss nor pyautogui available — screen capture disabled.")
        return "none"

    # ── Public API ────────────────────────────────────────────────────────────

    def capture(self, region: dict | None = None) -> "PIL.Image.Image | None":
        """Capture the screen and return a PIL Image."""
        if self._backend == "mss":
            return self._capture_mss(region)
        if self._backend == "pyautogui":
            return self._capture_pyautogui(region)
        return None

    def capture_and_save(self, filename: str | None = None) -> Path | None:
        """Capture and save to disk; return the saved path."""
        img = self.capture()
        if img is None:
            return None
        fname = filename or f"screen_{int(time.time())}.png"
        path  = self.save_dir / fname
        img.save(str(path))
        logger.debug("Screenshot saved: %s", path)
        return path

    def capture_bytes(self) -> bytes | None:
        """Return PNG bytes of the current screen."""
        img = self.capture()
        if img is None:
            return None
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def get_screen_text(self) -> str:
        """Basic description: resolution + active window title."""
        lines = []
        try:
            import pyautogui
            size = pyautogui.size()
            lines.append(f"Screen resolution: {size.width}x{size.height}")
        except Exception:
            pass
        try:
            import pygetwindow as gw
            active = gw.getActiveWindow()
            if active:
                lines.append(f"Active window: {active.title}")
        except Exception:
            pass
        return "\n".join(lines) if lines else "Screen capture unavailable."

    # ── Backends ──────────────────────────────────────────────────────────────

    def _capture_mss(self, region: dict | None):
        import mss
        import mss.tools
        from PIL import Image
        with mss.mss() as sct:
            monitor = region or sct.monitors[0]
            img_data = sct.grab(monitor)
            return Image.frombytes("RGB", img_data.size, img_data.bgra, "raw", "BGRX")

    def _capture_pyautogui(self, region: dict | None):
        import pyautogui
        if region:
            return pyautogui.screenshot(region=(
                region.get("left", 0), region.get("top", 0),
                region.get("width", 800), region.get("height", 600),
            ))
        return pyautogui.screenshot()
