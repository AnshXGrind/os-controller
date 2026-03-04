"""
voice/wake_word.py

Wake word detection layer.

The assistant remains passive until a recognised wake word is heard at
the start of an utterance.  Supported wake words are configurable.

Default wake words:
    "jarvis"     – primary wake word
    "computer"   – alternate
    "assistant"  – alternate

Usage in a listening loop:
    detector = WakeWordDetector()
    raw      = listener.listen_once()
    if detector.is_triggered(raw):
        command = detector.strip_wake_word(raw)
        # process command …
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default set of wake words (all lowercase)
DEFAULT_WAKE_WORDS: list[str] = [
    "jarvis",
    "computer",
    "assistant",
    "hey jarvis",
    "ok jarvis",
]


class WakeWordDetector:
    """
    Checks whether a recognised speech string starts with (or contains)
    a configured wake word.

    Args:
        wake_words: Iterable of trigger phrases (case-insensitive).
        require_prefix: If True, the wake word must appear at the
                        *beginning* of the utterance. If False, the
                        wake word may appear anywhere.
    """

    def __init__(
        self,
        wake_words:     list[str] = None,
        require_prefix: bool      = False,
    ):
        self._words: list[str] = [
            w.lower().strip()
            for w in (wake_words or DEFAULT_WAKE_WORDS)
        ]
        # Sort longest-first so multi-word phrases match before substrings
        self._words.sort(key=len, reverse=True)
        self.require_prefix = require_prefix

        logger.info(
            f"WakeWordDetector ready. "
            f"Trigger words: {self._words}  "
            f"(prefix-only={require_prefix})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_triggered(self, text: Optional[str]) -> bool:
        """
        Return True when *text* contains a wake word.

        Args:
            text: Recognised speech string (may be None).
        """
        if not text:
            return False
        normalised = text.lower().strip()
        for word in self._words:
            if self.require_prefix:
                if normalised.startswith(word):
                    return True
            else:
                if word in normalised:
                    return True
        return False

    def strip_wake_word(self, text: str) -> str:
        """
        Remove the leading wake word from *text* and return the remainder.

        Example:
            "jarvis open chrome"  →  "open chrome"
            "computer volume up"  →  "volume up"
        """
        normalised = text.lower().strip()
        for word in self._words:
            if normalised.startswith(word):
                # Strip the word + any following whitespace
                return text[len(word):].strip()
        return text.strip()

    def add_wake_word(self, word: str):
        """Dynamically add a new wake word at runtime."""
        w = word.lower().strip()
        if w not in self._words:
            self._words.append(w)
            self._words.sort(key=len, reverse=True)
            logger.info(f"WakeWordDetector: added '{w}'")

    @property
    def words(self) -> list[str]:
        """Read-only view of the current wake word list."""
        return list(self._words)
