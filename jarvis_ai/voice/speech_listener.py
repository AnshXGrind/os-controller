"""
voice/speech_listener.py

Microphone speech capture with multiple recognition back-ends.

Back-ends (in order of preference):
    1. Google Web Speech API  – online, highly accurate, no API key needed
    2. Vosk                   – offline, needs a downloaded model directory
    3. Whisper                – offline, OpenAI model via openai-whisper

The default back-end is Google.  Switch via the `backend` constructor arg.

Provides two modes:
    • listen_once()       – block until a single utterance is captured
    • continuous listen   – use listen_once() in a loop (see main.py)

Error handling:
    • UnknownValueError   → returns None (nothing understood)
    • RequestError        → logs warning, returns None
    • WaitTimeoutError    → returns None (silence timeout)
    • Microphone error    → raises RuntimeError at init time
"""

import logging
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Recognised back-end identifiers
BACKEND_GOOGLE = "google"
BACKEND_VOSK   = "vosk"
BACKEND_WHISPER = "whisper"


class SpeechListener:
    """
    Captures microphone audio and converts it to text.

    Args:
        backend:         Recognition backend ('google', 'vosk', 'whisper').
        language:        BCP-47 language tag (used by Google backend).
        energy_threshold:Ambient noise threshold for SpeechRecognition.
        phrase_timeout:  Max seconds of silence between words (phrase end).
        listen_timeout:  Max seconds to wait for speech to begin.
        vosk_model_path: Path to a Vosk model directory (vosk backend only).
        whisper_model:   Whisper model size: 'tiny','base','small','medium','large'.
    """

    def __init__(
        self,
        backend:          str   = BACKEND_GOOGLE,
        language:         str   = "en-US",
        energy_threshold: int   = 300,
        phrase_timeout:   float = 3.0,
        listen_timeout:   float = 5.0,
        vosk_model_path:  str   = "vosk-model-small-en-us",
        whisper_model:    str   = "base",
    ):
        self.backend          = backend
        self.language         = language
        self.phrase_timeout   = phrase_timeout
        self.listen_timeout   = listen_timeout
        self._vosk_model_path = vosk_model_path
        self._whisper_model   = whisper_model

        self._recognizer      = None
        self._vosk_rec        = None
        self._whisper_model_obj = None
        self._available       = True

        self._init_sr(energy_threshold)
        if backend == BACKEND_VOSK:
            self._init_vosk()
        elif backend == BACKEND_WHISPER:
            self._init_whisper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen_once(self) -> Optional[str]:
        """
        Block-listen for a single utterance and return the recognised text.

        Returns None on silence, unrecognised speech, or any error.
        """
        if not self._available:
            logger.error("SpeechListener unavailable – check installation.")
            return None

        sr = self._sr_lib
        try:
            mic = sr.Microphone()
        except Exception as exc:
            logger.error(f"Cannot open microphone: {exc}")
            return None

        try:
            with mic as source:
                logger.debug("Listening …")
                audio = self._recognizer.listen(
                    source,
                    timeout=self.listen_timeout,
                    phrase_time_limit=self.phrase_timeout,
                )
        except sr.WaitTimeoutError:
            return None
        except Exception as exc:
            logger.warning(f"listen error: {exc}")
            return None

        return self._recognise(audio)

    def calibrate(self, duration: float = 1.0):
        """
        Adjust the energy threshold to current ambient noise level.
        Call once at startup before entering the main loop.
        """
        if not self._available:
            return
        sr = self._sr_lib
        try:
            with sr.Microphone() as source:
                logger.info("Calibrating microphone to ambient noise …")
                self._recognizer.adjust_for_ambient_noise(source, duration=duration)
                logger.info(
                    f"Energy threshold set to {self._recognizer.energy_threshold:.0f}"
                )
        except Exception as exc:
            logger.warning(f"Calibration failed: {exc}")

    # ------------------------------------------------------------------
    # Recognition dispatch
    # ------------------------------------------------------------------

    def _recognise(self, audio) -> Optional[str]:
        """Route audio to the configured back-end and return text."""
        if self.backend == BACKEND_VOSK:
            return self._recognise_vosk(audio)
        if self.backend == BACKEND_WHISPER:
            return self._recognise_whisper(audio)
        return self._recognise_google(audio)

    def _recognise_google(self, audio) -> Optional[str]:
        sr = self._sr_lib
        try:
            text = self._recognizer.recognize_google(audio, language=self.language)
            logger.debug(f"Google heard: '{text}'")
            return text.lower().strip()
        except sr.UnknownValueError:
            return None
        except sr.RequestError as exc:
            logger.warning(f"Google Speech API error: {exc}")
            return None

    def _recognise_vosk(self, audio) -> Optional[str]:
        try:
            import json as _json
            wav_data = audio.get_wav_data()
            if self._vosk_rec.AcceptWaveform(wav_data):
                result = _json.loads(self._vosk_rec.Result())
                text   = result.get("text", "").strip()
                logger.debug(f"Vosk heard: '{text}'")
                return text or None
        except Exception as exc:
            logger.warning(f"Vosk recognition error: {exc}")
        return None

    def _recognise_whisper(self, audio) -> Optional[str]:
        try:
            import io, numpy as np
            wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
            arr = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            result = self._whisper_model_obj.transcribe(arr)
            text   = result.get("text", "").strip()
            logger.debug(f"Whisper heard: '{text}'")
            return text.lower() or None
        except Exception as exc:
            logger.warning(f"Whisper recognition error: {exc}")
        return None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_sr(self, energy_threshold: int):
        try:
            import speech_recognition as sr
            self._sr_lib     = sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold         = energy_threshold
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold           = 0.8
        except ImportError:
            logger.error(
                "speech_recognition not installed. "
                "Run: pip install SpeechRecognition pyaudio"
            )
            self._available = False

    def _init_vosk(self):
        try:
            from vosk import Model, KaldiRecognizer
            import wave
            model           = Model(self._vosk_model_path)
            self._vosk_rec  = KaldiRecognizer(model, 16000)
            logger.info(f"Vosk model loaded from '{self._vosk_model_path}'.")
        except ImportError:
            logger.warning("vosk not installed – falling back to Google.")
            self.backend = BACKEND_GOOGLE
        except Exception as exc:
            logger.warning(f"Vosk init failed ({exc}) – falling back to Google.")
            self.backend = BACKEND_GOOGLE

    def _init_whisper(self):
        try:
            import whisper
            logger.info(f"Loading Whisper model '{self._whisper_model}' …")
            self._whisper_model_obj = whisper.load_model(self._whisper_model)
            logger.info("Whisper model ready.")
        except ImportError:
            logger.warning("openai-whisper not installed – falling back to Google.")
            self.backend = BACKEND_GOOGLE
        except Exception as exc:
            logger.warning(f"Whisper init failed ({exc}) – falling back to Google.")
            self.backend = BACKEND_GOOGLE
