"""
utils/config.py

Single source of truth for all runtime configuration.

Values can be overridden via environment variables or the optional
jarvis_config.json file in the project root.

Priority (highest → lowest):
    1. Environment variable  (e.g. JARVIS_MODEL=llama3)
    2. jarvis_config.json    (user-editable file)
    3. Defaults below
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Default values ────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    # Ollama connection
    "OLLAMA_BASE_URL": "http://localhost:11434",
    "OLLAMA_MODEL":    "sammcj/qwen2.5-coder-7b-instruct:q8_0",  # installed model
    "OLLAMA_TIMEOUT":  30,           # seconds per HTTP request

    # Speech recognition
    "STT_BACKEND":          "google",    # google | vosk | whisper
    "STT_LANGUAGE":         "en-US",
    "STT_ENERGY_THRESHOLD": 300,
    "STT_LISTEN_TIMEOUT":   5,
    "STT_PHRASE_TIMEOUT":   4,
    "VOSK_MODEL_PATH":      "vosk-model-small-en-us",
    "WHISPER_MODEL":        "base",

    # Wake word
    "WAKE_WORDS":      ["jarvis", "computer", "assistant"],
    "REQUIRE_PREFIX":  False,

    # TTS
    "TTS_ENABLED":     True,
    "TTS_RATE":        175,
    "TTS_VOLUME":      0.9,

    # LLM intent parsing
    "USE_LLM":         True,     # use Ollama for intent parsing
    "LLM_FALLBACK_KW": True,     # fall back to keyword matching if LLM fails

    # Memory
    "HISTORY_FILE":    None,     # set to a path string to persist

    # Screen / cursor
    "SCREEN_WIDTH":    1920,
    "SCREEN_HEIGHT":   1080,

    # Action cooldown (seconds)
    "ACTION_COOLDOWN": 0.8,
}

# ── Load optional JSON config file ────────────────────────────────────────────
_CONFIG_FILE = Path(__file__).resolve().parents[2] / "jarvis_config.json"
_file_cfg: dict = {}
if _CONFIG_FILE.exists():
    try:
        _file_cfg = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        logger.info(f"Config: loaded overrides from '{_CONFIG_FILE}'")
    except Exception as exc:
        logger.warning(f"Config: could not parse '{_CONFIG_FILE}': {exc}")


def get(key: str, default: Any = None) -> Any:
    """
    Retrieve a configuration value.

    Resolution order:
        1. Environment variable:  JARVIS_<KEY>  (uppercased)
        2. jarvis_config.json
        3. Built-in default
        4. `default` argument

    Args:
        key:     Config key (case-insensitive).
        default: Value to return if nothing else matches.

    Returns:
        The resolved configuration value.
    """
    upper = key.upper()
    env_key = f"JARVIS_{upper}"

    # 1 – Environment variable
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return _cast(env_val, _DEFAULTS.get(upper))

    # 2 – JSON config file
    if key in _file_cfg:
        return _file_cfg[key]

    # 3 – Built-in default
    if upper in _DEFAULTS:
        return _DEFAULTS[upper]

    # 4 – Caller's default
    return default


def all_settings() -> dict:
    """Return a snapshot of all resolved settings (useful for debug logging)."""
    return {k: get(k) for k in _DEFAULTS}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cast(value: str, reference: Any) -> Any:
    """
    Cast an env-var string to the same type as the reference default.
    """
    if reference is None or isinstance(reference, str):
        return value
    if isinstance(reference, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(reference, int):
        try:
            return int(value)
        except ValueError:
            return value
    if isinstance(reference, float):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(reference, list):
        return [v.strip() for v in value.split(",")]
    return value
