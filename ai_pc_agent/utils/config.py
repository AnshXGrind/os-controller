"""ai_pc_agent/utils/config.py

Centralised configuration.
Override any default by:
  1. Setting an env var:  AI_AGENT_<KEY>=value
  2. Creating ai_agent_config.json next to main.py
"""

from __future__ import annotations
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_JSON_PATH = _ROOT / "ai_agent_config.json"

_DEFAULTS: dict[str, object] = {
    # ── Ollama ────────────────────────────────────────────────────────────────
    "OLLAMA_BASE_URL":      "http://localhost:11434",
    "OLLAMA_MODEL":         "sammcj/qwen2.5-coder-7b-instruct:q8_0",
    "OLLAMA_CODING_MODEL":  "sammcj/qwen2.5-coder-7b-instruct:q8_0",
    "OLLAMA_TIMEOUT":       60,

    # ── LLM behaviour ─────────────────────────────────────────────────────────
    "USE_LLM":              True,
    "LLM_FALLBACK_KW":      True,
    "LLM_MAX_TOKENS":       512,
    "LLM_TEMPERATURE":      0.2,

    # ── STT ───────────────────────────────────────────────────────────────────
    "STT_BACKEND":          "google",   # google | vosk | whisper
    "STT_LANGUAGE":         "en-US",
    "VOSK_MODEL_PATH":      "vosk-model-small-en-us",
    "WHISPER_MODEL":        "base",

    # ── Wake word ─────────────────────────────────────────────────────────────
    "WAKE_WORDS":           ["jarvis", "computer", "assistant"],

    # ── TTS ───────────────────────────────────────────────────────────────────
    "TTS_ENABLED":          True,
    "TTS_RATE":             175,
    "TTS_VOLUME":           0.9,

    # ── Memory ────────────────────────────────────────────────────────────────
    "HISTORY_FILE":         None,
    "SKILL_LIBRARY_FILE":   "skill_library.json",
    "MAX_CONTEXT_ITEMS":    20,

    # ── Vision ────────────────────────────────────────────────────────────────
    "SCREENSHOT_DIR":       "screenshots",
    "VISION_ENABLED":       True,

    # ── Safety ────────────────────────────────────────────────────────────────
    "ALLOW_SHUTDOWN":       False,    # must opt-in
    "ACTION_COOLDOWN":      0.5,
}

_json_cache: dict | None = None


def _load_json() -> dict:
    global _json_cache
    if _json_cache is None:
        if _JSON_PATH.exists():
            try:
                _json_cache = json.loads(_JSON_PATH.read_text())
            except Exception:
                _json_cache = {}
        else:
            _json_cache = {}
    return _json_cache


def get(key: str, default=None):
    """Resolve a config value: env → JSON file → built-in default."""
    env_key = f"AI_AGENT_{key.upper()}"
    if env_key in os.environ:
        val = os.environ[env_key]
        # Coerce type to match default
        ref = _DEFAULTS.get(key, default)
        if isinstance(ref, bool):
            return val.lower() in ("1", "true", "yes")
        if isinstance(ref, int):
            return int(val)
        if isinstance(ref, float):
            return float(val)
        if isinstance(ref, list):
            return val.split(",")
        return val
    json_val = _load_json().get(key)
    if json_val is not None:
        return json_val
    return _DEFAULTS.get(key, default)


def all_settings() -> dict:
    return {k: get(k) for k in _DEFAULTS}
