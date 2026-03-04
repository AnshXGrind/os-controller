"""
voice/tts_engine.py

Text-to-Speech (TTS) engine wrapper using pyttsx3 for fully offline synthesis.

Features:
    • Speaks response text through the system audio output
    • Configurable voice, rate, and volume
    • Non-blocking "speak_async" option
    • Graceful fallback (prints text) when pyttsx3 is unavailable

Usage:
    tts = TTSEngine()
    tts.speak("Opening Chrome")
    tts.speak_async("Processing your command")
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Wraps pyttsx3 for offline text-to-speech synthesis.

    Args:
        rate:     Words per minute (default 175 – natural conversational speed).
        volume:   Volume from 0.0 to 1.0 (default 0.9).
        voice_id: pyttsx3 voice index or ID string.
                  None = use system default.
        enabled:  Set False to silence all speech (log-only output).
    """

    def __init__(
        self,
        rate:     int            = 175,
        volume:   float          = 0.9,
        voice_id: Optional[str]  = None,
        enabled:  bool           = True,
    ):
        self.enabled  = enabled
        self._engine  = None
        self._lock    = threading.Lock()

        if not enabled:
            logger.info("TTSEngine: speech disabled (enabled=False).")
            return

        self._init(rate, volume, voice_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str):
        """
        Speak *text* synchronously (blocks until audio finishes).

        Args:
            text: The string to be spoken aloud.
        """
        if not text:
            return
        logger.info(f"Jarvis says: \"{text}\"")
        if not self.enabled or self._engine is None:
            print(f"[Jarvis]: {text}")
            return
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                logger.warning(f"TTS speak failed: {exc}")
                print(f"[Jarvis]: {text}")

    def speak_async(self, text: str):
        """
        Speak *text* in a background thread (non-blocking).

        Args:
            text: The string to be spoken aloud.
        """
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()

    def set_rate(self, rate: int):
        """Change speaking rate (words per minute)."""
        if self._engine:
            self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float):
        """Change speak volume (0.0 – 1.0)."""
        if self._engine:
            self._engine.setProperty("volume", max(0.0, min(1.0, volume)))

    def list_voices(self) -> list[str]:
        """Return a list of available voice IDs on this system."""
        if not self._engine:
            return []
        return [v.id for v in self._engine.getProperty("voices")]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init(self, rate: int, volume: float, voice_id: Optional[str]):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   rate)
            engine.setProperty("volume", volume)

            if voice_id is not None:
                engine.setProperty("voice", voice_id)
            else:
                # Prefer a female English voice if available (more Jarvis-like)
                voices = engine.getProperty("voices")
                for v in voices:
                    if "zira" in v.id.lower() or "female" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break

            self._engine = engine
            logger.info("TTSEngine ready (pyttsx3).")

        except ImportError:
            logger.error(
                "pyttsx3 not installed – TTS will print to console. "
                "Run: pip install pyttsx3"
            )
        except Exception as exc:
            logger.error(f"TTSEngine init failed: {exc} – TTS will print to console.")
