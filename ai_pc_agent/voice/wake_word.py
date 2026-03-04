"""ai_pc_agent/voice/wake_word.py

Wake-word gate — commands are only processed after the wake word is detected.
"""

from __future__ import annotations
from ai_pc_agent.utils import config
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.wake")


class WakeWordDetector:
    """Detect one of the configured wake words in a transcribed utterance."""

    def __init__(
        self,
        wake_words: list[str] | None = None,
        require_prefix: bool = False,
    ):
        raw = wake_words or config.get("WAKE_WORDS") or ["jarvis", "computer", "assistant"]
        if isinstance(raw, str):
            raw = [w.strip() for w in raw.split(",")]
        self.words:          list[str] = [w.lower().strip() for w in raw]
        self.require_prefix: bool      = require_prefix

    # ── Public API ────────────────────────────────────────────────────────────

    def is_triggered(self, text: str) -> bool:
        """Return True if any wake word appears in *text*."""
        tl = text.lower()
        return any(w in tl for w in self.words)

    def strip_wake_word(self, text: str) -> str:
        """Remove the wake word from the beginning of *text* and return the remainder."""
        tl = text.lower().strip()
        for w in sorted(self.words, key=len, reverse=True):  # longest first
            if tl.startswith(w):
                return text[len(w):].strip()
        return text.strip()

    def add_wake_word(self, word: str):
        w = word.lower().strip()
        if w not in self.words:
            self.words.append(w)

    def remove_wake_word(self, word: str):
        self.words = [w for w in self.words if w != word.lower().strip()]
