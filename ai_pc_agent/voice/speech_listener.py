"""ai_pc_agent/voice/speech_listener.py

Microphone audio capture + speech-to-text.
Supports three backends: google (online), vosk (offline), whisper (offline).
"""

from __future__ import annotations
import queue
import threading
from typing import Optional

from ai_pc_agent.utils import config
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.stt")

BACKEND_GOOGLE  = "google"
BACKEND_VOSK    = "vosk"
BACKEND_WHISPER = "whisper"


class SpeechListener:
    """Captures audio from the default microphone and returns transcribed text."""

    def __init__(
        self,
        backend: str | None = None,
        language: str | None = None,
        vosk_model_path: str | None = None,
        whisper_model: str | None = None,
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
        phrase_timeout: float = 3.0,
    ):
        self.backend         = (backend or config.get("STT_BACKEND", "google")).lower()
        self.language        = language or config.get("STT_LANGUAGE", "en-US")
        self.vosk_model_path = vosk_model_path or config.get("VOSK_MODEL_PATH")
        self.whisper_model   = whisper_model or config.get("WHISPER_MODEL", "base")
        self.energy_threshold = energy_threshold
        self.pause_threshold  = pause_threshold
        self.phrase_timeout   = phrase_timeout

        self._recogniser   = None
        self._mic          = None
        self._vosk_model   = None
        self._whisper_mdl  = None
        self._setup()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup(self):
        try:
            import speech_recognition as sr
            self._sr = sr
            self._recogniser = sr.Recognizer()
            self._recogniser.energy_threshold   = self.energy_threshold
            self._recogniser.pause_threshold    = self.pause_threshold
            self._recogniser.dynamic_energy_threshold = True
            self._mic = sr.Microphone()
        except ImportError:
            logger.error("speech_recognition not installed. Run: pip install speechrecognition pyaudio")
            raise

        if self.backend == BACKEND_VOSK:
            self._setup_vosk()
        elif self.backend == BACKEND_WHISPER:
            self._setup_whisper()

    def _setup_vosk(self):
        try:
            from vosk import Model, KaldiRecognizer
            self._vosk_model = Model(self.vosk_model_path)
            logger.info("Vosk model loaded from %s", self.vosk_model_path)
        except Exception as exc:
            logger.warning("Vosk setup failed (%s). Falling back to Google.", exc)
            self.backend = BACKEND_GOOGLE

    def _setup_whisper(self):
        try:
            import whisper
            self._whisper_mdl = whisper.load_model(self.whisper_model)
            logger.info("Whisper model '%s' loaded.", self.whisper_model)
        except Exception as exc:
            logger.warning("Whisper setup failed (%s). Falling back to Google.", exc)
            self.backend = BACKEND_GOOGLE

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, duration: float = 1.5):
        """Calibrate mic noise level."""
        logger.info("Calibrating microphone for %.1fs …", duration)
        try:
            with self._mic as source:
                self._recogniser.adjust_for_ambient_noise(source, duration=duration)
            logger.info("Calibration done. Energy threshold: %d", self._recogniser.energy_threshold)
        except Exception as exc:
            logger.warning("Calibration error: %s", exc)

    # ── Listening ─────────────────────────────────────────────────────────────

    def listen_once(self, timeout: float = 5.0, phrase_limit: float = 10.0) -> str | None:
        """Block until a phrase is captured, then return transcribed text (or None)."""
        try:
            with self._mic as source:
                logger.debug("Listening …")
                audio = self._recogniser.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit,
                )
            return self._recognise(audio)
        except self._sr.WaitTimeoutError:
            logger.debug("Listen timeout.")
            return None
        except Exception as exc:
            logger.warning("listen_once error: %s", exc)
            return None

    def _recognise(self, audio) -> str | None:
        if self.backend == BACKEND_VOSK:
            return self._recognise_vosk(audio)
        if self.backend == BACKEND_WHISPER:
            return self._recognise_whisper(audio)
        return self._recognise_google(audio)

    def _recognise_google(self, audio) -> str | None:
        try:
            text = self._recogniser.recognize_google(audio, language=self.language)
            return text.strip().lower()
        except self._sr.UnknownValueError:
            return None
        except self._sr.RequestError as exc:
            logger.warning("Google STT request error: %s", exc)
            return None

    def _recognise_vosk(self, audio) -> str | None:
        import json
        from vosk import KaldiRecognizer
        import wave, io
        wav_data = audio.get_wav_data()
        rec = KaldiRecognizer(self._vosk_model, 16000)
        rec.AcceptWaveform(wav_data)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        return text.lower() if text else None

    def _recognise_whisper(self, audio) -> str | None:
        import numpy as np
        import io
        wav_bytes = audio.get_wav_data(convert_rate=16000)
        audio_np  = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        result = self._whisper_mdl.transcribe(audio_np, language="en")
        text   = result.get("text", "").strip()
        return text.lower() if text else None
