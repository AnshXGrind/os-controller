"""ai_pc_agent/control/file_control.py

File system operations: open folders, create/delete/search files.
"""

from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path

from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.file")

FOLDER_PATHS: dict[str, str] = {
    "desktop":   os.path.expanduser("~/Desktop"),
    "downloads": os.path.expanduser("~/Downloads"),
    "documents": os.path.expanduser("~/Documents"),
    "pictures":  os.path.expanduser("~/Pictures"),
    "music":     os.path.expanduser("~/Music"),
    "videos":    os.path.expanduser("~/Videos"),
    "home":      os.path.expanduser("~"),
}


class FileControl:

    # ── Folders ───────────────────────────────────────────────────────────────

    def open_folder(self, name: str) -> bool:
        path = FOLDER_PATHS.get(name.lower().strip(), name)
        path = os.path.expandvars(path)
        try:
            os.startfile(path)
            logger.info("Opened folder: %s", path)
            return True
        except Exception as e:
            logger.error("open_folder '%s': %s", name, e)
            return False

    # ── Files ─────────────────────────────────────────────────────────────────

    def create_file(self, filename: str, folder: str = "desktop", content: str = "") -> bool:
        base = FOLDER_PATHS.get(folder.lower().strip(), folder)
        base = os.path.expandvars(base)
        path = Path(base) / filename
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info("Created file: %s", path)
            return True
        except Exception as e:
            logger.error("create_file '%s': %s", filename, e)
            return False

    def open_file(self, path: str) -> bool:
        path = os.path.expandvars(path)
        try:
            os.startfile(path)
            logger.info("Opened file: %s", path)
            return True
        except Exception as e:
            logger.error("open_file '%s': %s", path, e)
            return False

    def delete_file(self, path: str) -> bool:
        path = os.path.expandvars(path)
        try:
            p = Path(path)
            if not p.exists():
                logger.warning("delete_file: not found: %s", path)
                return False
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            logger.info("Deleted: %s", path)
            return True
        except Exception as e:
            logger.error("delete_file '%s': %s", path, e)
            return False

    def search_file(self, query: str, folder: str = "home") -> list[str]:
        base = FOLDER_PATHS.get(folder.lower().strip(), folder)
        base = Path(os.path.expandvars(base))
        query = query.lower()
        results: list[str] = []
        try:
            for p in base.rglob("*"):
                if query in p.name.lower():
                    results.append(str(p))
                if len(results) >= 20:
                    break
        except Exception as e:
            logger.error("search_file '%s': %s", query, e)
        logger.info("search_file '%s': %d result(s)", query, len(results))
        return results

    def read_file(self, path: str, max_chars: int = 4000) -> str:
        try:
            return Path(os.path.expandvars(path)).read_text(encoding="utf-8")[:max_chars]
        except Exception as e:
            logger.error("read_file: %s", e)
            return ""

    def write_file(self, path: str, content: str) -> bool:
        try:
            p = Path(os.path.expandvars(path))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            logger.info("Wrote file: %s", p)
            return True
        except Exception as e:
            logger.error("write_file: %s", e)
            return False

    def list_folder(self, folder: str = "desktop") -> list[str]:
        base = FOLDER_PATHS.get(folder.lower().strip(), folder)
        base = Path(os.path.expandvars(base))
        try:
            return [str(p.name) for p in base.iterdir()]
        except Exception as e:
            logger.error("list_folder: %s", e)
            return []
