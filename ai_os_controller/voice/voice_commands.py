"""
voice/voice_commands.py

Threaded speech recognition module using the SpeechRecognition library.

The microphone is listened to in a background thread so that the main
camera-processing loop is never blocked waiting for audio.

Recognised command keywords → normalised command strings:

    "next"        → "next"
    "previous"    → "previous"
    "like"        → "like"
    "pause"       → "pause"
    "play"        → "play"
    "volume up"   → "volume_up"
    "volume down" → "volume_down"
    "scroll up"   → "scroll_up"
    "scroll down" → "scroll_down"
    "stop"        → "stop"
"""

import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Map spoken phrases to canonical command identifiers
COMMAND_MAP: dict[str, str] = {
    "next":        "next",
    "previous":    "previous",
    "prev":        "previous",
    "back":        "previous",
    "like":        "like",
    "pause":       "pause",
    "play":        "play",
    "volume up":   "volume_up",
    "volume down": "volume_down",
    "louder":      "volume_up",
    "quieter":     "volume_down",
    "scroll up":   "scroll_up",
    "scroll down": "scroll_down",
    "stop":        "stop",
}


class VoiceController:
    """
    Continuously listens for voice commands in a background thread and
    exposes the latest recognised command via get_command().

    Usage:
        vc = VoiceController()
        vc.start()
        ...
        cmd = vc.get_command()   # non-blocking; returns None if nothing new
        ...
        vc.stop()
    """

    def __init__(self, language: str = "en-US", energy_threshold: int = 300):
        """
        Args:
            language:         BCP-47 language tag passed to Google's API.
            energy_threshold: Microphone ambient noise threshold for
                              speech_recognition.
        """
        self.language         = language
        self.energy_threshold = energy_threshold

        self._command_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available = True   # set False if sr/pyaudio unavailable

        self._init_recognizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "VoiceController":
        """Spawn the background listening thread."""
        if not self._available:
            logger.warning("VoiceController is unavailable – speech disabled.")
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("VoiceController: listening thread started.")
        return self

    def get_command(self) -> Optional[str]:
        """
        Non-blocking pop of the latest voice command from the queue.

        Returns:
            Command string if one is ready, otherwise None.
        """
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Signal the background thread to stop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("VoiceController: stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_recognizer(self):
        """Initialise speech_recognition objects; mark unavailable on import error."""
        try:
            import speech_recognition as sr   # noqa: F401
            self._sr = sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.dynamic_energy_threshold = True
        except ImportError:
            logger.error(
                "speech_recognition / pyaudio is not installed. "
                "Voice commands will be disabled. "
                "Run: pip install SpeechRecognition pyaudio"
            )
            self._available = False

    def _listen_loop(self):
        """
        Background thread: continuously listen for phrases and push
        normalised commands onto the queue.
        """
        sr = self._sr

        try:
            mic = sr.Microphone()
        except Exception as exc:
            logger.error(f"Cannot access microphone: {exc}")
            return

        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)

        while not self._stop_event.is_set():
            try:
                with mic as source:
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)

                text = self.recognizer.recognize_google(
                    audio, language=self.language
                ).lower().strip()

                logger.debug(f"VoiceController heard: '{text}'")

                command = self._parse(text)
                if command:
                    logger.info(f"VoiceController command: '{command}'")
                    self._command_queue.put(command)

            except sr.WaitTimeoutError:
                pass   # silence – just loop again
            except sr.UnknownValueError:
                pass   # speech not understood
            except sr.RequestError as exc:
                logger.warning(f"Google Speech API error: {exc}")
            except Exception as exc:
                logger.warning(f"VoiceController unexpected error: {exc}")

    @staticmethod
    def _parse(text: str) -> Optional[str]:
        """
        Match the recognised text against COMMAND_MAP.

        Checks for exact key matches as well as substring containment so
        that incidental words around the keyword still trigger the command.
        """
        for keyword, command in COMMAND_MAP.items():
            if keyword in text:
                return command
        return None
