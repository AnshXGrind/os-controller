"""
core/intent_parser.py

Converts free-form speech text into a structured (intent, value) tuple.

Two parsing strategies are provided:

Strategy 1 – Keyword matching (default, offline, instant)
    Scans the utterance for known keyword phrases from INTENT_KEYWORDS.
    Returns the first matching intent.  App / folder / query values are
    extracted with simple pattern matching.

Strategy 2 – LLM-based parsing (optional, requires OpenAI API key)
    Sends the utterance to an LLM with a structured prompt and parses
    the JSON response for higher naturalness handling.
    e.g. "Could you turn the sound up a bit?" → ("volume_up", "")

The parser always returns a tuple:
    (intent_string: str, value_string: str)
    ("unknown", "") is returned when no intent can be matched.
"""

import logging
import re
from typing import Optional

from jarvis_ai.utils.command_map import (
    INTENT_KEYWORDS,
    APP_PATHS,
    APP_ALIASES,
    FOLDER_PATHS,
)

logger = logging.getLogger(__name__)

# All known app names (union of paths and aliases)
_KNOWN_APPS: set[str] = set(APP_PATHS.keys()) | set(APP_ALIASES.keys())

# All known folder names
_KNOWN_FOLDERS: set[str] = set(FOLDER_PATHS.keys())


class IntentParser:
    """
    Maps a recognised speech string to a (intent, value) pair.

    Args:
        use_llm:    Enable LLM-based natural language parsing as a
                    fallback when keyword matching fails.
        openai_key: OpenAI API key (required only when use_llm=True).
        llm_model:  Model name string (default "gpt-3.5-turbo").
    """

    def __init__(
        self,
        use_llm:    bool = False,
        openai_key: str  = "",
        llm_model:  str  = "gpt-3.5-turbo",
    ):
        self.use_llm    = use_llm
        self._oai_key   = openai_key
        self._llm_model = llm_model

        # Pre-sort keyword lists longest-first to avoid substring shadowing
        self._kw: dict[str, list[str]] = {
            intent: sorted(phrases, key=len, reverse=True)
            for intent, phrases in INTENT_KEYWORDS.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, text: str) -> tuple[str, str]:
        """
        Parse a raw utterance into (intent, value).

        Args:
            text: Lowercase, stripped speech text.

        Returns:
            Tuple of (intent_string, value_string).
            value_string is "" when the intent has no associated value.
        """
        if not text:
            return "unknown", ""

        normalised = text.lower().strip()

        # ── Keyword matching ──────────────────────────────────────────
        result = self._keyword_parse(normalised)
        if result[0] != "unknown":
            return result

        # ── LLM fallback ──────────────────────────────────────────────
        if self.use_llm:
            result = self._llm_parse(normalised)
            if result[0] != "unknown":
                return result

        logger.debug(f"IntentParser: no intent matched for '{text}'")
        return "unknown", ""

    # ------------------------------------------------------------------
    # Strategy 1 – keyword matching
    # ------------------------------------------------------------------

    def _keyword_parse(self, text: str) -> tuple[str, str]:
        """Scan *text* for known keyword phrases and extract a value."""

        # ── App open/close – check before generic keywords
        #    "open chrome", "launch vscode", "close spotify" …
        for action, triggers in [
            ("open_app",  self._kw["open_app"]),
            ("close_app", self._kw["close_app"]),
        ]:
            for kw in triggers:
                if kw in text:
                    app = self._extract_app(text, kw)
                    if app:
                        return action, app

        # ── Folder open
        for trigger in self._kw.get("open_folder", []):
            if trigger in text:
                folder = self._extract_folder(text)
                return "open_folder", folder or "documents"

        # ── Search / query  (extract the search term)
        for trigger in self._kw.get("search_google", []):
            if trigger in text:
                query = self._extract_after(text, trigger)
                return "search_google", query

        # ── File search
        for trigger in self._kw.get("search_file", []):
            if trigger in text:
                query = self._extract_after(text, trigger)
                return "search_file", query

        # ── All remaining single-keyword intents
        for intent, keywords in self._kw.items():
            if intent in ("open_app", "close_app", "open_folder",
                          "search_google", "search_file"):
                continue    # already handled above
            for kw in keywords:
                if kw in text:
                    return intent, ""

        return "unknown", ""

    # ------------------------------------------------------------------
    # Value extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_app(text: str, trigger: str) -> Optional[str]:
        """
        Find an app name after the trigger word.

        Example: "open chrome browser" → "chrome"
        """
        after = text.split(trigger, 1)[-1].strip()
        # Check alias first, then direct key
        for word in after.split():
            if word in APP_ALIASES:
                return APP_ALIASES[word]
            if word in APP_PATHS:
                return word
        # Partial match
        for word in after.split():
            for key in APP_PATHS:
                if word in key:
                    return key
        return after.split()[0] if after.split() else None

    @staticmethod
    def _extract_folder(text: str) -> Optional[str]:
        """Return the first matching folder keyword found in *text*."""
        for folder in _KNOWN_FOLDERS:
            if folder in text:
                return folder
        return None

    @staticmethod
    def _extract_after(text: str, trigger: str) -> str:
        """Return everything after the first occurrence of *trigger*."""
        parts = text.split(trigger, 1)
        return parts[-1].strip() if len(parts) > 1 else ""

    # ------------------------------------------------------------------
    # Strategy 2 – LLM parsing (optional)
    # ------------------------------------------------------------------

    def _llm_parse(self, text: str) -> tuple[str, str]:
        """
        Use an LLM to map natural language to a structured intent.

        Returns ("unknown", "") on any error.
        """
        try:
            import openai, json as _json
            openai.api_key = self._oai_key

            known_intents = list(INTENT_KEYWORDS.keys())
            system_prompt = (
                "You are a voice command interpreter for a PC assistant. "
                "Convert the user's utterance into a JSON object with exactly "
                "two fields: 'intent' (one of the known intents listed below) "
                "and 'value' (string, empty if not applicable). "
                f"Known intents: {known_intents}"
            )
            user_prompt = (
                f"Command: \"{text}\"\n"
                "Respond ONLY with valid JSON. Example: {\"intent\": \"volume_up\", \"value\": \"\"}"
            )

            response = openai.ChatCompletion.create(
                model    = self._llm_model,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature = 0,
                max_tokens  = 60,
            )
            raw   = response.choices[0].message.content.strip()
            data  = _json.loads(raw)
            intent = str(data.get("intent", "unknown")).lower()
            value  = str(data.get("value",  "")).lower()
            logger.info(f"IntentParser LLM: '{text}' → ({intent}, '{value}')")
            return intent, value

        except ImportError:
            logger.warning("openai not installed – LLM fallback disabled.")
        except Exception as exc:
            logger.warning(f"IntentParser LLM error: {exc}")

        return "unknown", ""
