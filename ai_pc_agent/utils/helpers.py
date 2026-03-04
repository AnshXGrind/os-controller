"""ai_pc_agent/utils/helpers.py

Shared utility functions used throughout the project.
"""

from __future__ import annotations
import os
import re
import sys
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Any


# ── String helpers ────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert text to a safe filename fragment."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "_", text)[:60]


def truncate(text: str, max_len: int = 100, suffix: str = "…") -> str:
    return text if len(text) <= max_len else text[:max_len - len(suffix)] + suffix


def clean_llm_output(text: str) -> str:
    """Strip markdown fences and excess whitespace from LLM responses."""
    text = re.sub(r"```(?:json|python|bash)?", "", text)
    text = text.strip("`").strip()
    return text


def extract_json(text: str) -> str | None:
    """Extract first {...} JSON block from a string."""
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    return m.group() if m else None


def extract_code_blocks(text: str) -> list[str]:
    """Extract all ```...``` code blocks."""
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


# ── System helpers ────────────────────────────────────────────────────────────

def is_windows() -> bool:
    return sys.platform == "win32"


def is_admin() -> bool:
    if is_windows():
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return os.geteuid() == 0


def run_cmd(cmd: list[str] | str, timeout: int = 10) -> tuple[bool, str]:
    """Run a shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=isinstance(cmd, str),
            capture_output=True, text=True, timeout=timeout,
        )
        output = (result.stdout or result.stderr or "").strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as exc:
        return False, str(exc)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Time helpers ──────────────────────────────────────────────────────────────

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def elapsed_str(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


# ── Hash helpers ──────────────────────────────────────────────────────────────

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


# ── App detection ─────────────────────────────────────────────────────────────

def cmd_exists(name: str) -> bool:
    """Check if a command-line tool is available on PATH."""
    from shutil import which
    return which(name) is not None


def ollama_running(base_url: str = "http://localhost:11434") -> bool:
    try:
        import requests
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"
