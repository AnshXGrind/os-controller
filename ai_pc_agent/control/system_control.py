"""ai_pc_agent/control/system_control.py

OS-level control: volume, brightness, power management, screen lock, screenshots.
"""

from __future__ import annotations
import subprocess
import time

import pyautogui

from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.system")


class SystemControl:

    # ── Volume ────────────────────────────────────────────────────────────────

    def volume_up(self, steps: int = 2) -> bool:
        try:
            for _ in range(steps):
                pyautogui.press("volumeup")
            logger.info("Volume up ×%d", steps)
            return True
        except Exception as e:
            logger.error("volume_up: %s", e); return False

    def volume_down(self, steps: int = 2) -> bool:
        try:
            for _ in range(steps):
                pyautogui.press("volumedown")
            logger.info("Volume down ×%d", steps)
            return True
        except Exception as e:
            logger.error("volume_down: %s", e); return False

    def mute(self) -> bool:
        try:
            pyautogui.press("volumemute")
            logger.info("Mute toggled")
            return True
        except Exception as e:
            logger.error("mute: %s", e); return False

    def set_volume(self, level: int) -> bool:
        """Set volume to *level* percent (Windows only via nircmd or PowerShell)."""
        level = max(0, min(100, level))
        try:
            script = (
                f"$obj = New-Object -com 'wscript.shell'; "
                f"$vol = [math]::Round({level} / 100 * 65535); "
                "Add-Type -AssemblyName System.Windows.Forms; "
                "[System.Windows.Forms.SendKeys]::SendWait('')"
            )
            # Use nircmd if available, otherwise skip
            result = subprocess.run(
                ["nircmd.exe", "setsysvolume", str(int(level / 100 * 65535))],
                capture_output=True, timeout=5,
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning("set_volume: %s", e); return False

    # ── Brightness (Windows via WMI) ──────────────────────────────────────────

    def brightness_up(self, step: int = 10) -> bool:
        return self._adjust_brightness(+step)

    def brightness_down(self, step: int = 10) -> bool:
        return self._adjust_brightness(-step)

    def _adjust_brightness(self, delta: int) -> bool:
        try:
            import wmi
            c   = wmi.WMI(namespace="wmi")
            obj = c.WmiMonitorBrightnessMethods()[0]
            cur = c.WmiMonitorBrightness()[0].CurrentBrightness
            new = max(0, min(100, cur + delta))
            obj.WmiSetBrightness(new, 0)
            logger.info("Brightness %+d → %d%%", delta, new)
            return True
        except Exception as e:
            logger.warning("brightness adjust: %s", e)
            return False

    # ── Power ─────────────────────────────────────────────────────────────────

    def shutdown(self, delay: int = 5) -> bool:
        from ai_pc_agent.utils import config
        if not config.get("ALLOW_SHUTDOWN", False):
            logger.warning("Shutdown blocked (ALLOW_SHUTDOWN=False)")
            return False
        logger.info("Shutting down in %ds …", delay)
        subprocess.Popen(["shutdown", "/s", "/t", str(delay)])
        return True

    def restart(self, delay: int = 5) -> bool:
        from ai_pc_agent.utils import config
        if not config.get("ALLOW_SHUTDOWN", False):
            logger.warning("Restart blocked (ALLOW_SHUTDOWN=False)")
            return False
        logger.info("Restarting in %ds …", delay)
        subprocess.Popen(["shutdown", "/r", "/t", str(delay)])
        return True

    def lock_screen(self) -> bool:
        try:
            import ctypes
            ctypes.windll.user32.LockWorkStation()
            logger.info("Screen locked")
            return True
        except Exception as e:
            logger.error("lock_screen: %s", e); return False

    def cancel_shutdown(self) -> bool:
        subprocess.Popen(["shutdown", "/a"])
        return True

    # ── Screenshot ────────────────────────────────────────────────────────────

    def screenshot(self, path: str | None = None) -> str | None:
        try:
            p = path or f"screenshot_{int(time.time())}.png"
            pyautogui.screenshot(p)
            logger.info("Screenshot: %s", p)
            return p
        except Exception as e:
            logger.error("screenshot: %s", e); return None
