"""
core/assistant_engine.py

The central coordinator of Jarvis AI.

Orchestration flow per cycle:
    1. SpeechListener.listen_once()    → raw text string
    2. WakeWordDetector.is_triggered() → gate check
    3. WakeWordDetector.strip_wake_word() → clean command
    4. IntentParser.parse()            → (intent, value)
    5. _dispatch(intent, value)        → invoke the correct controller
    6. TTSEngine.speak()               → vocal response
    7. CommandHistory.add()            → record for memory

Priority for ambiguous commands:
    system commands (shutdown/restart) > app control > browser > file

The engine runs synchronously; the main loop in main.py drives
the frame rate / polling interval.
"""

import logging
import time
from typing import Optional

from jarvis_ai.voice.speech_listener  import SpeechListener
from jarvis_ai.voice.wake_word        import WakeWordDetector
from jarvis_ai.voice.tts_engine       import TTSEngine
from jarvis_ai.core.intent_parser     import IntentParser
from jarvis_ai.control.system_control import SystemControl
from jarvis_ai.control.app_control    import AppControl
from jarvis_ai.control.browser_control import BrowserControl
from jarvis_ai.control.file_control   import FileControl
from jarvis_ai.memory.command_history import CommandHistory
from jarvis_ai.utils.command_map      import RESPONSE_TEMPLATES

logger = logging.getLogger(__name__)


class AssistantEngine:
    """
    Drives the full Jarvis voice command lifecycle.

    Args:
        listener:   SpeechListener instance.
        wake:       WakeWordDetector instance.
        tts:        TTSEngine instance.
        parser:     IntentParser instance.
        system:     SystemControl instance.
        apps:       AppControl instance.
        browser:    BrowserControl instance.
        files:      FileControl instance.
        history:    CommandHistory instance.
        idle_response: Text to speak when wake word heard but no intent matched.
    """

    def __init__(
        self,
        listener:      SpeechListener,
        wake:          WakeWordDetector,
        tts:           TTSEngine,
        parser:        IntentParser,
        system:        SystemControl,
        apps:          AppControl,
        browser:       BrowserControl,
        files:         FileControl,
        history:       CommandHistory,
        idle_response: str = "I didn't catch that. Please try again.",
    ):
        self.listener      = listener
        self.wake          = wake
        self.tts           = tts
        self.parser        = parser
        self.system        = system
        self.apps          = apps
        self.browser       = browser
        self.files         = files
        self.history       = history
        self.idle_response = idle_response

        self._running  = False
        self._sleeping = False    # true after "stop listening" command

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_once(self) -> bool:
        """
        Execute one full listen → parse → act → respond cycle.

        Returns:
            False if the engine should stop (user said "stop"), True otherwise.
        """
        # ── 1. Listen ─────────────────────────────────────────────────
        raw = self.listener.listen_once()
        if not raw:
            return True    # silence – keep running

        logger.info(f"Heard: '{raw}'")

        # ── 2. Wake word check ────────────────────────────────────────
        if self._sleeping:
            # Re-activate only when a wake word is heard
            if self.wake.is_triggered(raw):
                self._sleeping = False
                self.tts.speak("I'm awake. How can I help?")
            return True

        if not self.wake.is_triggered(raw):
            logger.debug("No wake word – ignoring.")
            return True

        # ── 3. Strip wake word to get the command payload ──────────────────
        command = self.wake.strip_wake_word(raw)
        logger.info(f"Command: '{command}'")

        if not command:
            self.tts.speak("Yes? I'm listening.")
            return True

        # ── 4. Parse intent ───────────────────────────────────────────
        intent, value = self.parser.parse(command)
        logger.info(f"Intent: ({intent}, '{value}')")

        # ── 5. Dispatch action ────────────────────────────────────────
        success = self._dispatch(intent, value, raw)

        # ── 6. TTS response ───────────────────────────────────────────
        response = self._build_response(intent, value)
        self.tts.speak(response)

        # ── 7. Record in history ──────────────────────────────────────
        self.history.add(raw_text=raw, intent=intent, value=value, success=success)

        # ── 8. Check for stop signal ──────────────────────────────────
        if intent == "stop":
            self._sleeping = True
            return True    # stay in loop but go to sleep mode

        return True

    def run(self):
        """Start the blocking main loop (runs until KeyboardInterrupt)."""
        self._running = True
        self.tts.speak("Jarvis is online. Waiting for your command.")
        logger.info("AssistantEngine: running. Say a wake word to begin.")
        try:
            while self._running:
                self.run_once()
        except KeyboardInterrupt:
            logger.info("AssistantEngine: KeyboardInterrupt – stopping.")
        finally:
            self._running = False
            self.tts.speak("Jarvis shutting down. Goodbye.")

    def stop(self):
        """Signal the run loop to exit cleanly."""
        self._running = False

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, intent: str, value: str, raw: str) -> bool:
        """
        Route intent to the correct controller method.

        Returns True if the action was executed without error, False otherwise.
        """
        try:
            # ── Meta / memory ────────────────────────────────────────────
            if intent == "repeat_command":
                last = self.history.last_successful()
                if last:
                    logger.info(f"Repeating: ({last.intent}, '{last.value}')")
                    return self._dispatch(last.intent, last.value, last.raw_text)
                return False

            if intent == "show_history":
                lines = self.history.summary_lines(10)
                self.tts.speak(f"You have {len(self.history)} commands in history.")
                for line in lines:
                    print(line)
                return True

            if intent == "help":
                self.tts.speak(
                    "I can open and close apps, control volume and brightness, "
                    "navigate the browser, manage files, and control system power."
                )
                return True

            if intent == "stop":
                return True   # handled by run_once

            # ── System control ───────────────────────────────────────────
            if intent == "volume_up":
                self.system.volume_up();  return True
            if intent == "volume_down":
                self.system.volume_down(); return True
            if intent == "mute":
                self.system.mute();       return True
            if intent == "brightness_up":
                self.system.brightness_up();   return True
            if intent == "brightness_down":
                self.system.brightness_down(); return True
            if intent == "shutdown":
                self.system.shutdown(); return True
            if intent == "restart":
                self.system.restart();  return True
            if intent == "lock_screen":
                self.system.lock_screen(); return True
            if intent == "screenshot":
                self.system.screenshot(); return True

            # ── App control ───────────────────────────────────────────────
            if intent == "open_app":
                return self.apps.open_app(value)
            if intent == "close_app":
                return self.apps.close_app(value)

            # ── Browser control ───────────────────────────────────────────
            if intent == "new_tab":
                self.browser.new_tab();       return True
            if intent == "close_tab":
                self.browser.close_tab();     return True
            if intent == "next_tab":
                self.browser.next_tab();      return True
            if intent == "previous_tab":
                self.browser.previous_tab();  return True
            if intent == "scroll_up":
                self.browser.scroll_up();     return True
            if intent == "scroll_down":
                self.browser.scroll_down();   return True
            if intent == "go_back":
                self.browser.go_back();       return True
            if intent == "go_forward":
                self.browser.go_forward();    return True
            if intent == "refresh":
                self.browser.refresh();       return True
            if intent == "search_google":
                self.browser.search(value);   return True

            # ── Window management ─────────────────────────────────────────
            if intent == "minimize_window":
                self.browser.minimize_window();  return True
            if intent == "maximize_window":
                self.browser.maximize_window();  return True
            if intent == "close_window":
                self.browser.close_window();     return True
            if intent == "switch_window":
                self.browser.switch_window();    return True

            # ── File control ──────────────────────────────────────────────
            if intent == "open_folder":
                return self.files.open_folder(value or "documents")
            if intent == "search_file":
                return self.files.search_file(value)
            if intent == "create_file":
                path = self.files.create_file()
                return bool(path)

            # ── Unknown ───────────────────────────────────────────────────
            logger.debug(f"AssistantEngine: unhandled intent '{intent}'")
            return False

        except Exception as exc:
            logger.error(f"AssistantEngine._dispatch error: {exc}")
            return False

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response(intent: str, value: str) -> str:
        """
        Generate a natural-language TTS response from the response template.
        """
        template = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["unknown"])
        try:
            return template.format(value=value) if "{value}" in template else template
        except (KeyError, IndexError):
            return template
