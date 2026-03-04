"""
core/assistant_brain.py

AI-powered command interpreter using the local Ollama LLM.

Converts free-form natural language utterances into structured
(intent, value) tuples by asking the LLM to classify the command.

The LLM is given a strict system prompt that instructs it to respond
ONLY with a JSON object.  A keyword-matching fallback is used when:
    • Ollama is unavailable
    • The LLM returns a non-JSON response
    • The response confidence is low

Supported intents mirror the keyword-based IntentParser so both
parsers share the same downstream CommandRouter.
"""

import json
import logging
import re
from typing import Optional

from jarvis_ai.ai.ollama_client  import OllamaClient
from jarvis_ai.core.intent_parser import IntentParser  # keyword fallback
from jarvis_ai.utils import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt for the LLM
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_LIST = [
    "open_app", "close_app",
    "volume_up", "volume_down", "mute",
    "brightness_up", "brightness_down",
    "shutdown", "restart", "lock_screen", "screenshot",
    "new_tab", "close_tab", "next_tab", "previous_tab",
    "scroll_up", "scroll_down",
    "go_back", "go_forward", "refresh",
    "search_google",
    "minimize_window", "maximize_window", "close_window", "switch_window",
    "open_folder", "create_file", "search_file",
    "repeat_command", "show_history", "help", "stop",
    "unknown",
]

_SYSTEM_PROMPT = f"""You are a strict PC voice assistant command classifier.

Your ONLY job is to convert a user voice command into a JSON object.

Rules:
- Respond with ONLY a JSON object. No explanation, no markdown, no extra text.
- Use exactly one of these intents: {_INTENT_LIST}
- "value" must be a short string (app name, search query, folder name) or empty string "".
- If you cannot match the command, use intent "unknown".

Examples:
User: "open chrome" → {{"intent": "open_app", "value": "chrome"}}
User: "search python tutorial" → {{"intent": "search_google", "value": "python tutorial"}}
User: "turn up the volume" → {{"intent": "volume_up", "value": ""}}
User: "shut down the computer" → {{"intent": "shutdown", "value": ""}}
User: "open my documents folder" → {{"intent": "open_folder", "value": "documents"}}
User: "what can you do" → {{"intent": "help", "value": ""}}
"""


class AssistantBrain:
    """
    Hybrid intent parser: tries Ollama LLM first, falls back to keyword matching.

    Args:
        ollama:       OllamaClient instance (or None to skip LLM).
        use_llm:      Enable LLM parsing (overrides config if supplied).
        fallback_kw:  Fall back to keyword matching when LLM fails.
    """

    def __init__(
        self,
        ollama:      OllamaClient = None,
        use_llm:     bool         = None,
        fallback_kw: bool         = None,
    ):
        self.ollama      = ollama
        self.use_llm     = use_llm     if use_llm     is not None else config.get("USE_LLM", True)
        self.fallback_kw = fallback_kw if fallback_kw is not None else config.get("LLM_FALLBACK_KW", True)

        # Keyword parser is always available as a safety net
        self._kw_parser = IntentParser(use_llm=False)

        # LLM availability flag (checked lazily on first call)
        self._llm_ok: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, command: str) -> tuple[str, str]:
        """
        Parse a voice command string into (intent, value).

        Pipeline:
            1. Try Ollama LLM  →  parse JSON response
            2. If LLM fails / unavailable → keyword fallback

        Args:
            command: Stripped, lowercase command text (wake word removed).

        Returns:
            (intent_str, value_str)
        """
        if not command:
            return "unknown", ""

        # ── LLM path ─────────────────────────────────────────────────
        if self.use_llm and self.ollama and self._llm_available():
            result = self._llm_parse(command)
            if result and result[0] != "unknown":
                return result

        # ── Keyword fallback ──────────────────────────────────────────
        if self.fallback_kw:
            result = self._kw_parser.parse(command)
            if result[0] != "unknown":
                return result

        return "unknown", ""

    # ------------------------------------------------------------------
    # LLM parsing
    # ------------------------------------------------------------------

    def _llm_parse(self, command: str) -> Optional[tuple[str, str]]:
        """
        Ask the LLM to classify the command.  Returns None on any failure.
        """
        prompt = f'User command: "{command}"\nRespond with JSON only.'

        try:
            data = self.ollama.ask_json(prompt, system=_SYSTEM_PROMPT)
        except Exception as exc:
            logger.warning(f"AssistantBrain LLM call failed: {exc}")
            return None

        if not data:
            logger.debug(f"AssistantBrain: LLM returned no JSON for '{command}'")
            return None

        intent = str(data.get("intent", "unknown")).lower().strip()
        value  = str(data.get("value",  "")).lower().strip()

        # Validate intent is in the allowed list
        if intent not in _INTENT_LIST:
            logger.debug(
                f"AssistantBrain: LLM returned unknown intent '{intent}' – ignoring."
            )
            return None

        logger.info(f"AssistantBrain LLM → ({intent}, '{value}')  cmd='{command}'")
        return intent, value

    # ------------------------------------------------------------------
    # LLM availability check (cached)
    # ------------------------------------------------------------------

    def _llm_available(self) -> bool:
        if self._llm_ok is None:
            self._llm_ok = self.ollama.is_available()
            if not self._llm_ok:
                logger.warning(
                    "Ollama is not available. "
                    "Running in keyword-only mode. "
                    "Start the server with: ollama serve"
                )
        return self._llm_ok

    def reset_availability(self):
        """Force a fresh availability check on the next parse() call."""
        self._llm_ok = None
