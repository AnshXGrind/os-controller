"""
core/command_router.py

Maps parsed (intent, value) pairs to concrete controller method calls.

This is the single dispatch table for the entire system.  Adding a new
capability only requires:
    1. Adding the intent string to the INTENT_LIST in assistant_brain.py
    2. Adding a handler branch here in CommandRouter.route()

All controller objects are injected at construction time so the router
has no hard dependencies on specific implementations.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CommandRouter:
    """
    Routes intents to the correct controller action.

    Args:
        system:   SystemControl instance.
        apps:     AppControl instance.
        browser:  BrowserControl instance.
        files:    FileControl instance.
        history:  CommandHistory instance (for repeat / show_history).
        tts:      TTSEngine instance (for inline responses like help/history).
    """

    def __init__(self, system, apps, browser, files, history=None, tts=None):
        self.system  = system
        self.apps    = apps
        self.browser = browser
        self.files   = files
        self.history = history
        self.tts     = tts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, intent: str, value: str, raw: str = "") -> tuple[bool, str]:
        """
        Execute the action for the given intent.

        Args:
            intent: Intent string from the parser / LLM.
            value:  Optional value string (app name, query, path …).
            raw:    Original spoken text (used for repeat command).

        Returns:
            (success: bool, response_text: str)
            response_text is the TTS string Jarvis should speak.
        """
        logger.debug(f"CommandRouter: intent='{intent}'  value='{value}'")

        try:
            return self._dispatch(intent, value, raw)
        except Exception as exc:
            logger.error(f"CommandRouter error for intent='{intent}': {exc}")
            return False, "Sorry, something went wrong while executing that command."

    # ------------------------------------------------------------------
    # Dispatch table
    # ------------------------------------------------------------------

    def _dispatch(self, intent: str, value: str, raw: str) -> tuple[bool, str]:
        # ── Meta ─────────────────────────────────────────────────────────────
        if intent == "help":
            msg = (
                "I can open and close apps, control volume and brightness, "
                "navigate the browser, manage files, and control power. "
                "Just say Jarvis followed by what you need."
            )
            return True, msg

        if intent == "show_history":
            if self.history:
                lines = self.history.summary_lines(5)
                count = len(self.history)
                if self.tts:
                    self.tts.speak(f"Your last {min(5, count)} commands:")
                for line in lines:
                    print(line)
                return True, f"Showing your last {min(5, count)} commands."
            return False, "No history available."

        if intent == "repeat_command":
            if self.history:
                last = self.history.last_successful()
                if last:
                    logger.info(f"Repeating: ({last.intent}, '{last.value}')")
                    return self._dispatch(last.intent, last.value, last.raw_text)
            return False, "No previous command to repeat."

        if intent == "stop":
            return True, "Going to sleep. Say my name to wake me."

        if intent == "unknown":
            return False, "Sorry, I didn't understand that command. Please try again."

        # ── Volume ────────────────────────────────────────────────────────────
        if intent == "volume_up":
            self.system.volume_up()
            return True, "Turning up the volume."

        if intent == "volume_down":
            self.system.volume_down()
            return True, "Turning down the volume."

        if intent == "mute":
            self.system.mute()
            return True, "Muting audio."

        # ── Brightness ───────────────────────────────────────────────────────
        if intent == "brightness_up":
            self.system.brightness_up()
            return True, "Increasing brightness."

        if intent == "brightness_down":
            self.system.brightness_down()
            return True, "Decreasing brightness."

        # ── Power ─────────────────────────────────────────────────────────────
        if intent == "shutdown":
            self.tts and self.tts.speak("Shutting down the computer. Goodbye!")
            self.system.shutdown()
            return True, ""

        if intent == "restart":
            self.tts and self.tts.speak("Restarting the system.")
            self.system.restart()
            return True, ""

        if intent == "lock_screen":
            self.system.lock_screen()
            return True, "Locking the screen."

        if intent == "screenshot":
            path = self.system.screenshot()
            return True, f"Screenshot saved." if path else "Screenshot failed."

        # ── Apps ──────────────────────────────────────────────────────────────
        if intent == "open_app":
            if not value:
                return False, "Which app would you like me to open?"
            ok = self.apps.open_app(value)
            return ok, f"Opening {value}." if ok else f"I couldn't find {value}."

        if intent == "close_app":
            if not value:
                return False, "Which app would you like me to close?"
            ok = self.apps.close_app(value)
            return ok, f"Closing {value}." if ok else f"Couldn't close {value}."

        # ── Browser tabs ──────────────────────────────────────────────────────
        if intent == "new_tab":
            self.browser.new_tab()
            return True, "Opening a new tab."

        if intent == "close_tab":
            self.browser.close_tab()
            return True, "Closing the tab."

        if intent == "next_tab":
            self.browser.next_tab()
            return True, "Switching to the next tab."

        if intent == "previous_tab":
            self.browser.previous_tab()
            return True, "Switching to the previous tab."

        # ── Browser navigation ────────────────────────────────────────────────
        if intent == "scroll_down":
            self.browser.scroll_down()
            return True, "Scrolling down."

        if intent == "scroll_up":
            self.browser.scroll_up()
            return True, "Scrolling up."

        if intent == "go_back":
            self.browser.go_back()
            return True, "Going back."

        if intent == "go_forward":
            self.browser.go_forward()
            return True, "Going forward."

        if intent == "refresh":
            self.browser.refresh()
            return True, "Refreshing."

        if intent == "search_google":
            q = value or "latest news"
            self.browser.search(q)
            return True, f"Searching for {q}."

        # ── Window management ─────────────────────────────────────────────────
        if intent == "minimize_window":
            self.browser.minimize_window()
            return True, "Minimizing the window."

        if intent == "maximize_window":
            self.browser.maximize_window()
            return True, "Maximizing the window."

        if intent == "close_window":
            self.browser.close_window()
            return True, "Closing the window."

        if intent == "switch_window":
            self.browser.switch_window()
            return True, "Switching windows."

        # ── Files ─────────────────────────────────────────────────────────────
        if intent == "open_folder":
            folder = value or "documents"
            ok = self.files.open_folder(folder)
            return ok, f"Opening {folder} folder." if ok else f"Couldn't find {folder}."

        if intent == "create_file":
            path = self.files.create_file()
            return bool(path), "Creating a new file." if path else "File creation failed."

        if intent == "search_file":
            ok = self.files.search_file(value)
            return ok, f"Searching for {value}." if ok else "File search unavailable."

        # ── Fallthrough ───────────────────────────────────────────────────────
        logger.warning(f"CommandRouter: unhandled intent '{intent}'")
        return False, f"I don't know how to {intent.replace('_', ' ')} yet."
