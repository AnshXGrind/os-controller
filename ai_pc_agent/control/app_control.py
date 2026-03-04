"""ai_pc_agent/control/app_control.py

Open, close, and focus desktop applications.
"""

from __future__ import annotations
import subprocess
import time

import psutil

from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.app")

# ── App path registry ─────────────────────────────────────────────────────────

APP_PATHS: dict[str, str] = {
    "chrome":       r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "firefox":      r"C:\Program Files\Mozilla Firefox\firefox.exe",
    "edge":         r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "vscode":       r"C:\Users\%USERNAME%\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "notepad":      "notepad.exe",
    "notepad++":    r"C:\Program Files\Notepad++\notepad++.exe",
    "explorer":     "explorer.exe",
    "calculator":   "calc.exe",
    "spotify":      r"C:\Users\%USERNAME%\AppData\Roaming\Spotify\Spotify.exe",
    "discord":      r"C:\Users\%USERNAME%\AppData\Local\Discord\Update.exe",
    "slack":        r"C:\Users\%USERNAME%\AppData\Local\slack\slack.exe",
    "terminal":     "wt.exe",       # Windows Terminal
    "cmd":          "cmd.exe",
    "powershell":   "powershell.exe",
    "paint":        "mspaint.exe",
    "word":         r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
    "excel":        r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
    "teams":        r"C:\Users\%USERNAME%\AppData\Local\Microsoft\Teams\current\Teams.exe",
    "zoom":         r"C:\Users\%USERNAME%\AppData\Roaming\Zoom\bin\Zoom.exe",
    "vlc":          r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    "obs":          r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
}

APP_ALIASES: dict[str, str] = {
    "google chrome": "chrome",
    "visual studio code": "vscode",
    "vs code": "vscode",
    "code": "vscode",
    "windows terminal": "terminal",
    "microsoft edge": "edge",
    "windows explorer": "explorer",
    "file explorer": "explorer",
}


class AppControl:
    """Open, close, and focus applications."""

    def _resolve(self, name: str) -> str:
        key = name.lower().strip()
        key = APP_ALIASES.get(key, key)
        return APP_PATHS.get(key, key)  # fallback: use as-is

    # ── Open ──────────────────────────────────────────────────────────────────

    def open_app(self, name: str) -> bool:
        path = self._resolve(name)
        try:
            import os
            path = os.path.expandvars(path)
            subprocess.Popen([path], shell=True)
            logger.info("Opened: %s  (%s)", name, path)
            return True
        except Exception as e:
            logger.error("open_app '%s': %s", name, e)
            return False

    # ── Close ─────────────────────────────────────────────────────────────────

    def close_app(self, name: str) -> bool:
        key         = name.lower().strip()
        key         = APP_ALIASES.get(key, key)
        exe_name    = key.split("\\")[-1].split("/")[-1]
        if not exe_name.endswith(".exe"):
            exe_name += ".exe"
        killed = False
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                if proc.info["name"] and proc.info["name"].lower() == exe_name.lower():
                    proc.kill()
                    killed = True
                    logger.info("Killed process: %s (pid=%d)", proc.info["name"], proc.info["pid"])
            except Exception:
                pass
        if not killed:
            logger.warning("close_app: no process found for '%s'", name)
        return killed

    # ── Focus ─────────────────────────────────────────────────────────────────

    def focus_app(self, name: str) -> bool:
        try:
            import pygetwindow as gw
            key = name.lower().strip()
            key = APP_ALIASES.get(key, key)
            windows = [w for w in gw.getAllWindows()
                       if key in (w.title or "").lower()]
            if windows:
                win = windows[0]
                if win.isMinimized:
                    win.restore()
                win.activate()
                logger.info("Focused window: %s", win.title)
                return True
            logger.warning("focus_app: no window found for '%s'", name)
            return False
        except Exception as e:
            logger.error("focus_app '%s': %s", name, e)
            return False

    # ── Status ────────────────────────────────────────────────────────────────

    def is_running(self, name: str) -> bool:
        key      = APP_ALIASES.get(name.lower().strip(), name.lower().strip())
        exe_name = key.split("\\")[-1]
        if not exe_name.endswith(".exe"):
            exe_name += ".exe"
        return any(
            p.info["name"] and p.info["name"].lower() == exe_name.lower()
            for p in psutil.process_iter(["name"])
        )

    def list_running(self) -> list[str]:
        return list({
            p.info["name"].lower().replace(".exe", "")
            for p in psutil.process_iter(["name"])
            if p.info["name"]
        })
