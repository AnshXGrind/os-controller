"""
jarvis_ai/main.py

Entry point for Jarvis AI – Local AI Voice Controlled PC Assistant.

Uses a locally-running Ollama model as the reasoning engine for
natural-language command interpretation.

Usage (from workspace root):
    python -m jarvis_ai.main

    # Show Ollama status and available models:
    python -m jarvis_ai.main --check

    # Use a different Ollama model:
    python -m jarvis_ai.main --model llama3

    # Use offline Vosk STT:
    python -m jarvis_ai.main --backend vosk --vosk-model vosk-model-small-en-us

    # Use Whisper STT:
    python -m jarvis_ai.main --backend whisper --whisper-model base

    # Disable wake-word gate (every utterance is a command):
    python -m jarvis_ai.main --no-wake

    # Silent (no TTS – print responses only):
    python -m jarvis_ai.main --silent

    # Persist command history:
    python -m jarvis_ai.main --history-file jarvis_history.json

Pre-requisites:
    1. Ollama running:  ollama serve
    2. A model pulled:  ollama pull llama3
                    or: ollama pull sammcj/qwen2.5-coder-7b-instruct:q8_0

Press  Ctrl+C  to quit.
"""

import argparse
import logging
import sys
import time

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis.main")

# ── Module imports ────────────────────────────────────────────────────────────
from jarvis_ai.voice.speech_listener    import SpeechListener
from jarvis_ai.voice.wake_word          import WakeWordDetector
from jarvis_ai.voice.tts_engine         import TTSEngine
from jarvis_ai.ai.ollama_client         import OllamaClient
from jarvis_ai.core.assistant_brain     import AssistantBrain
from jarvis_ai.core.command_router      import CommandRouter
from jarvis_ai.control.system_control   import SystemControl
from jarvis_ai.control.app_control      import AppControl
from jarvis_ai.control.browser_control  import BrowserControl
from jarvis_ai.control.file_control     import FileControl
from jarvis_ai.memory.command_history   import CommandHistory
from jarvis_ai.utils                    import config


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "jarvis_ai",
        description = "Jarvis AI – Local LLM Voice Controlled PC Assistant",
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # Diagnostics
    p.add_argument("--check", action="store_true",
                   help="Print Ollama status and available models then exit")

    # Ollama / LLM
    p.add_argument("--model", default=None,
                   help="Ollama model name (default from config)")
    p.add_argument("--ollama-url", default=None,
                   help="Ollama server base URL (default: http://localhost:11434)")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable Ollama; use keyword-only intent parsing")

    # Speech recognition
    p.add_argument("--backend", choices=["google", "vosk", "whisper"],
                   default=config.get("STT_BACKEND", "google"),
                   help="Speech recognition backend (default: google)")
    p.add_argument("--language", default=config.get("STT_LANGUAGE", "en-US"),
                   help="BCP-47 language tag for Google STT (default: en-US)")
    p.add_argument("--vosk-model", default=config.get("VOSK_MODEL_PATH"),
                   help="Path to Vosk model directory")
    p.add_argument("--whisper-model",
                   choices=["tiny", "base", "small", "medium", "large"],
                   default=config.get("WHISPER_MODEL", "base"),
                   help="Whisper model size (default: base)")

    # Wake word
    p.add_argument("--no-wake", action="store_true",
                   help="Disable wake-word gate (all utterances processed)")
    p.add_argument("--wake-words", nargs="+",
                   default=config.get("WAKE_WORDS"),
                   help="Custom wake word(s) (default: jarvis computer assistant)")

    # TTS
    p.add_argument("--silent", action="store_true",
                   help="Disable TTS (print responses only)")
    p.add_argument("--tts-rate", type=int, default=config.get("TTS_RATE", 175),
                   help="TTS speaking rate in WPM (default: 175)")

    # Memory
    p.add_argument("--history-file", default=config.get("HISTORY_FILE"),
                   help="JSON file path to persist command history")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Ollama diagnostics helper
# ─────────────────────────────────────────────────────────────────────────────

def _check_ollama(client: OllamaClient):
    import requests
    print("\n── Ollama Diagnostics ────────────────────────────────")
    try:
        resp = requests.get(f"{client.base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        print(f"  Server : {client.base_url}  ✓ reachable")
        print(f"  Models ({len(models)}):")
        for m in models:
            active = " ← active" if m["name"] == client.model else ""
            print(f"    • {m['name']}{active}")
    except Exception:
        print(f"  Server : {client.base_url}  ✗ NOT reachable")
        print("  → Start with:  ollama serve")
        print("  → Pull model:  ollama pull llama3")
    print("─────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _build_parser().parse_args()

    logger.info("═" * 60)
    logger.info("  Jarvis AI – Local LLM Voice Controller  (Ollama)")
    logger.info("═" * 60)

    # ── Ollama client ─────────────────────────────────────────────────────────
    ollama = OllamaClient(
        base_url = args.ollama_url,
        model    = args.model,
    )

    if args.check:
        _check_ollama(ollama)
        sys.exit(0)

    logger.info(f"Ollama model : {ollama.model}")
    logger.info(f"Ollama server: {ollama.base_url}")

    if not args.no_llm:
        available = ollama.is_available()
        if not available:
            logger.warning(
                "Ollama is not reachable or the model is not available. "
                "Falling back to keyword-only mode. "
                "Start the server with: ollama serve"
            )
    else:
        available = False
        logger.info("LLM disabled by --no-llm flag (keyword mode only).")

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts = TTSEngine(rate=args.tts_rate, enabled=not args.silent)

    # ── Speech listener ───────────────────────────────────────────────────────
    logger.info(f"STT backend: {args.backend}")
    listener = SpeechListener(
        backend         = args.backend,
        language        = args.language,
        vosk_model_path = args.vosk_model,
        whisper_model   = args.whisper_model,
    )
    listener.calibrate()

    # ── Wake word ─────────────────────────────────────────────────────────────
    wake = WakeWordDetector(wake_words=args.wake_words, require_prefix=False)
    if args.no_wake:
        wake.is_triggered    = lambda _: True
        wake.strip_wake_word = lambda t: t
        logger.info("Wake-word gate DISABLED.")

    # ── AI brain (LLM + keyword fallback) ────────────────────────────────────
    brain = AssistantBrain(
        ollama      = ollama if not args.no_llm else None,
        use_llm     = not args.no_llm,
        fallback_kw = True,
    )

    # ── Controllers ───────────────────────────────────────────────────────────
    system  = SystemControl()
    apps    = AppControl()
    browser = BrowserControl()
    files   = FileControl()

    # ── Memory ────────────────────────────────────────────────────────────────
    history = CommandHistory(persist_file=args.history_file)

    # ── Command router ────────────────────────────────────────────────────────
    router = CommandRouter(
        system  = system,
        apps    = apps,
        browser = browser,
        files   = files,
        history = history,
        tts     = tts,
    )

    # ── Announce readiness ────────────────────────────────────────────────────
    logger.info(f"Wake words : {wake.words}")
    logger.info("All modules ready.  Speak a wake word to begin.")
    logger.info("Press Ctrl+C to quit.")
    logger.info("─" * 60)
    tts.speak("Jarvis online. Waiting for your command.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    sleeping = False    # True after "stop" command; re-wake on next wake word

    try:
        while True:
            # ── 1. Listen ─────────────────────────────────────────────────────
            raw = listener.listen_once()
            if not raw:
                continue    # silence or recognition error

            logger.info(f"Heard: '{raw}'")

            # ── 2. Wake word gate ─────────────────────────────────────────────
            if sleeping:
                if wake.is_triggered(raw):
                    sleeping = False
                    tts.speak("I'm awake. How can I help?")
                continue

            if not wake.is_triggered(raw):
                logger.debug("No wake word – ignoring.")
                continue

            command = wake.strip_wake_word(raw)
            if not command:
                tts.speak("Yes? I'm listening.")
                continue

            # ── 3. Parse intent via LLM / fallback ───────────────────────────
            logger.info(f"Parsing: '{command}'")
            intent, value = brain.parse(command)
            logger.info(f"Intent: ({intent}, '{value}')")

            # ── 4. Route and execute ──────────────────────────────────────────
            success, response = router.route(intent, value, raw=raw)

            # ── 5. Speak response ─────────────────────────────────────────────
            if response:
                tts.speak(response)

            # ── 6. Record in memory ───────────────────────────────────────────
            history.add(raw_text=raw, intent=intent, value=value, success=success)

            # ── 7. Sleep mode ─────────────────────────────────────────────────
            if intent == "stop":
                sleeping = True

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        tts.speak("Jarvis shutting down. Goodbye.")
        logger.info("Jarvis AI stopped.")


if __name__ == "__main__":
    main()
