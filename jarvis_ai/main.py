"""
jarvis_ai/main.py

Entry point for Jarvis AI – Voice Controlled PC Assistant.

Usage:
    python -m jarvis_ai.main

    # Skip wake-word gate (every utterance is treated as a command):
    python -m jarvis_ai.main --no-wake

    # Use offline Vosk STT instead of Google:
    python -m jarvis_ai.main --backend vosk --vosk-model vosk-model-small-en-us

    # Use Whisper STT:
    python -m jarvis_ai.main --backend whisper --whisper-model base

    # Silent mode (no TTS – print responses instead):
    python -m jarvis_ai.main --silent

    # Persist command history to file:
    python -m jarvis_ai.main --history-file jarvis_history.json

    # Enable LLM-based natural language intent parsing:
    python -m jarvis_ai.main --use-llm --openai-key sk-...

Press  Ctrl+C  to quit.
"""

import argparse
import logging
import sys

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis.main")

# ── Module imports ────────────────────────────────────────────────────────────
from jarvis_ai.voice.speech_listener   import SpeechListener, BACKEND_GOOGLE
from jarvis_ai.voice.wake_word         import WakeWordDetector
from jarvis_ai.voice.tts_engine        import TTSEngine
from jarvis_ai.core.intent_parser      import IntentParser
from jarvis_ai.core.assistant_engine   import AssistantEngine
from jarvis_ai.control.system_control  import SystemControl
from jarvis_ai.control.app_control     import AppControl
from jarvis_ai.control.browser_control import BrowserControl
from jarvis_ai.control.file_control    import FileControl
from jarvis_ai.memory.command_history  import CommandHistory


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jarvis_ai",
        description="Jarvis AI – Voice Controlled PC Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Speech recognition
    p.add_argument("--backend", choices=["google", "vosk", "whisper"],
                   default="google",
                   help="Speech recognition backend (default: google)")
    p.add_argument("--language", default="en-US",
                   help="BCP-47 language tag for Google STT (default: en-US)")
    p.add_argument("--vosk-model", default="vosk-model-small-en-us",
                   help="Path to Vosk model directory")
    p.add_argument("--whisper-model",
                   choices=["tiny", "base", "small", "medium", "large"],
                   default="base",
                   help="Whisper model size (default: base)")

    # Wake word
    p.add_argument("--no-wake", action="store_true",
                   help="Disable wake-word gate (all utterances processed)")
    p.add_argument("--wake-words", nargs="+",
                   default=["jarvis", "computer", "assistant"],
                   help="Custom wake word(s) (default: jarvis computer assistant)")

    # TTS
    p.add_argument("--silent", action="store_true",
                   help="Disable TTS speech output (print responses only)")
    p.add_argument("--tts-rate", type=int, default=175,
                   help="TTS speaking rate in WPM (default: 175)")

    # LLM
    p.add_argument("--use-llm", action="store_true",
                   help="Enable LLM-based intent parsing as fallback")
    p.add_argument("--openai-key", default="",
                   help="OpenAI API key (required with --use-llm)")

    # History
    p.add_argument("--history-file", default=None,
                   help="JSON file path to persist command history")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _build_parser().parse_args()

    logger.info("═" * 60)
    logger.info("  Jarvis AI – Voice Controlled PC Assistant")
    logger.info("═" * 60)

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts = TTSEngine(rate=args.tts_rate, enabled=not args.silent)

    # ── Speech listener ───────────────────────────────────────────────────────
    logger.info(f"Speech backend: {args.backend}")
    listener = SpeechListener(
        backend         = args.backend,
        language        = args.language,
        vosk_model_path = args.vosk_model,
        whisper_model   = args.whisper_model,
    )
    listener.calibrate()

    # ── Wake word ─────────────────────────────────────────────────────────────
    wake = WakeWordDetector(
        wake_words     = args.wake_words,
        require_prefix = False,
    )
    if args.no_wake:
        # Override: always "triggered"
        wake.is_triggered   = lambda _: True     # type: ignore[method-assign]
        wake.strip_wake_word = lambda t: t        # type: ignore[method-assign]
        logger.info("Wake-word gate DISABLED – processing all utterances.")

    # ── Intent parser ─────────────────────────────────────────────────────────
    parser = IntentParser(
        use_llm    = args.use_llm,
        openai_key = args.openai_key,
    )

    # ── Controllers ───────────────────────────────────────────────────────────
    system  = SystemControl()
    apps    = AppControl()
    browser = BrowserControl()
    files   = FileControl()

    # ── Memory ────────────────────────────────────────────────────────────────
    history = CommandHistory(persist_file=args.history_file)

    # ── Engine ────────────────────────────────────────────────────────────────
    engine = AssistantEngine(
        listener = listener,
        wake     = wake,
        tts      = tts,
        parser   = parser,
        system   = system,
        apps     = apps,
        browser  = browser,
        files    = files,
        history  = history,
    )

    logger.info("All modules ready.  Speak a wake word to begin.")
    logger.info(f"Wake words: {wake.words}")
    logger.info("Press Ctrl+C to quit.")
    logger.info("─" * 60)

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Jarvis AI stopped.")


if __name__ == "__main__":
    main()
