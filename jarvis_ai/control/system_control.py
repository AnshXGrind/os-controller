"""
control/system_control.py

OS-level system control: volume, brightness, power management.

All actions use pyautogui key presses or Windows shell commands so the
module works without any elevated privileges (except shutdown/restart
which naturally require admin rights on some systems).
"""

import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)


class SystemControl:
    """
    Controls core OS functions: audio, power, display.

    Args:
        volume_step:      Number of volume key taps per call.
        brightness_step:  WMI brightness increment per call (Windows only).
    """

    def __init__(self, volume_step: int = 2, brightness_step: int = 10):
        self.volume_step     = volume_step
        self.brightness_step = brightness_step

        # Lazy-import pyautogui so the module still loads if it's absent
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            self._pag      = pyautogui
            self._pag_ok   = True
        except ImportError:
            logger.error("pyautogui not installed – system key controls disabled.")
            self._pag_ok = False

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    def volume_up(self):
        """Raise system volume by `volume_step` key taps."""
        logger.debug("SystemControl: volume_up")
        if self._pag_ok:
            for _ in range(self.volume_step):
                self._pag.press("volumeup")
                time.sleep(0.05)

    def volume_down(self):
        """Lower system volume by `volume_step` key taps."""
        logger.debug("SystemControl: volume_down")
        if self._pag_ok:
            for _ in range(self.volume_step):
                self._pag.press("volumedown")
                time.sleep(0.05)

    def mute(self):
        """Toggle system mute."""
        logger.debug("SystemControl: mute")
        if self._pag_ok:
            self._pag.press("volumemute")

    # ------------------------------------------------------------------
    # Brightness  (Windows WMI – no-op on non-Windows or missing WMI)
    # ------------------------------------------------------------------

    def brightness_up(self):
        """Increase display brightness (Windows WMI, no-op elsewhere)."""
        logger.debug("SystemControl: brightness_up")
        self._adjust_brightness(+self.brightness_step)

    def brightness_down(self):
        """Decrease display brightness (Windows WMI, no-op elsewhere)."""
        logger.debug("SystemControl: brightness_down")
        self._adjust_brightness(-self.brightness_step)

    def _adjust_brightness(self, delta: int):
        """Read current brightness, clamp, and set via WMI."""
        try:
            import wmi
            c       = wmi.WMI(namespace="wmi")
            methods = c.WmiMonitorBrightnessMethods()[0]
            monitor = c.WmiMonitorBrightness()[0]
            current = monitor.CurrentBrightness
            new_val = max(0, min(100, current + delta))
            methods.WmiSetBrightness(new_val, 0)
            logger.info(f"Brightness set to {new_val}%")
        except ImportError:
            logger.warning("wmi module not available – brightness control disabled.")
        except Exception as exc:
            logger.warning(f"Brightness adjustment failed: {exc}")

    # ------------------------------------------------------------------
    # Power management
    # ------------------------------------------------------------------

    def shutdown(self):
        """Shut down the PC immediately."""
        logger.warning("SystemControl: SHUTDOWN initiated")
        os.system("shutdown /s /t 1")

    def restart(self):
        """Restart the PC immediately."""
        logger.warning("SystemControl: RESTART initiated")
        os.system("shutdown /r /t 1")

    def lock_screen(self):
        """Lock the Windows workstation."""
        logger.info("SystemControl: lock_screen")
        os.system("rundll32.exe user32.dll,LockWorkStation")

    def sleep(self):
        """Put the system to sleep (requires powershell)."""
        logger.info("SystemControl: sleep")
        subprocess.run(
            ["powershell", "-command",
             "Add-Type -Assembly System.Windows.Forms; "
             "[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"],
            check=False,
        )

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def screenshot(self, save_path: str = None) -> str:
        """
        Take a full-screen screenshot.

        Args:
            save_path: File path to save the image. Defaults to
                       timestamped PNG in the current directory.
        Returns:
            Path of the saved file.
        """
        if not self._pag_ok:
            logger.warning("pyautogui unavailable – cannot take screenshot.")
            return ""
        if save_path is None:
            ts        = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"screenshot_{ts}.png"
        img = self._pag.screenshot()
        img.save(save_path)
        logger.info(f"Screenshot saved to '{save_path}'")
        return save_path
