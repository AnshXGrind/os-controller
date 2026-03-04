"""ai_pc_agent/voice/tts_engine.py

Text-to-Speech via pyttsx3 (fully offline).
"""

from __future__ import annotations
import threading

from ai_pc_agent.utils import config
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.tts")


class TTSEngine:
    """Speak text using pyttsx3.  Thread-safe via a dedicated worker thread."""

    def __init__(
        self,
        rate: int | None = None,
        volume: float | None = None,
        enabled: bool | None = None,
    ):
        self.rate    = rate    or int(config.get("TTS_RATE", 175))
        self.volume  = volume  or float(config.get("TTS_VOLUME", 0.9))
        self.enabled = enabled if enabled is not None else bool(config.get("TTS_ENABLED", True))
        self._engine = None
        self._lock   = threading.Lock()
        if self.enabled:
            self._init_engine()

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   self.rate)
            self._engine.setProperty("volume", self.volume)
            voices = self._engine.getProperty("voices")
            if voices:
                self._engine.setProperty("voice", voices[0].id)
            logger.debug("TTS engine initialised. Rate=%d, Volume=%.1f", self.rate, self.volume)
        except Exception as exc:
            logger.warning("TTS init failed: %s. Speech disabled.", exc)
            self.enabled = False

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Speak *text* synchronously (blocks until done)."""
        if not self.enabled or not text:
            logger.info("[TTS] %s", text)
            return
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except RuntimeError:
                # Engine busy — reinitialise
                self._init_engine()
                if self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()

    def speak_async(self, text: str):
        """Speak *text* in a background thread (non-blocking)."""
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()

    def set_rate(self, rate: int):
        self.rate = rate
        if self._engine:
            self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))
        if self._engine:
            self._engine.setProperty("volume", self.volume)
