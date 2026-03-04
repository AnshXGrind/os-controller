"""ai_pc_agent/core/command_router.py

Map (intent, value) pairs to concrete controller method calls.
Returns (success: bool, response_text: str).
"""

from __future__ import annotations
import time

from ai_pc_agent.control.system_control  import SystemControl
from ai_pc_agent.control.app_control     import AppControl
from ai_pc_agent.control.browser_control import BrowserControl
from ai_pc_agent.control.file_control    import FileControl
from ai_pc_agent.control.keyboard_mouse  import KeyboardMouse
from ai_pc_agent.control.vscode_control  import VSCodeControl
from ai_pc_agent.memory.command_history  import CommandHistory
from ai_pc_agent.memory.skill_library    import SkillLibrary
from ai_pc_agent.utils.logger            import get_logger

logger = get_logger("agent.router")


class CommandRouter:
    """Route structured intents to the correct controller method."""

    def __init__(
        self,
        system:   SystemControl,
        apps:     AppControl,
        browser:  BrowserControl,
        files:    FileControl,
        kb_mouse: KeyboardMouse,
        vscode:   VSCodeControl,
        history:  CommandHistory,
        skills:   SkillLibrary,
        tts=None,
    ):
        self.system   = system
        self.apps     = apps
        self.browser  = browser
        self.files    = files
        self.kb       = kb_mouse
        self.vscode   = vscode
        self.history  = history
        self.skills   = skills
        self.tts      = tts
        self._script_cache: dict[str, str] = {}

    # ── Public dispatch ───────────────────────────────────────────────────────

    def route(self, intent: str, value: str, raw: str = "") -> tuple[bool, str]:
        """Execute intent and return (success, response_text)."""
        logger.info("Route: intent=%s  value=%r", intent, value)
        handler = self._DISPATCH.get(intent)
        if handler:
            try:
                return handler(self, value, raw)
            except Exception as exc:
                logger.error("Handler error [%s]: %s", intent, exc)
                return False, f"Error executing {intent}: {exc}"
        return False, f"Unknown intent: {intent}"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ok(self, msg: str) -> tuple[bool, str]:
        return True, msg

    def _err(self, msg: str) -> tuple[bool, str]:
        return False, msg

    def _say(self, text: str):
        if self.tts:
            self.tts.speak_async(text)

    # ══════════════════════════════════════════════════════════════════════════
    # Handler methods (must accept self, value, raw)
    # ══════════════════════════════════════════════════════════════════════════

    def _open_app(self, value, _):
        ok = self.apps.open_app(value)
        return self._ok(f"Opening {value}") if ok else self._err(f"Could not open {value}")

    def _close_app(self, value, _):
        ok = self.apps.close_app(value)
        return self._ok(f"Closed {value}") if ok else self._err(f"Could not close {value}")

    def _focus_app(self, value, _):
        ok = self.apps.focus_app(value)
        return self._ok(f"Focused {value}") if ok else self._err(f"Could not focus {value}")

    def _volume_up(self, value, _):
        steps = int(value) if value.isdigit() else 2
        self.system.volume_up(steps)
        return self._ok("Volume increased")

    def _volume_down(self, value, _):
        steps = int(value) if value.isdigit() else 2
        self.system.volume_down(steps)
        return self._ok("Volume decreased")

    def _mute(self, *_):
        self.system.mute()
        return self._ok("Muted")

    def _brightness_up(self, *_):
        self.system.brightness_up()
        return self._ok("Brightness increased")

    def _brightness_down(self, *_):
        self.system.brightness_down()
        return self._ok("Brightness decreased")

    def _shutdown(self, *_):
        self._say("Shutting down the computer")
        time.sleep(1)
        ok = self.system.shutdown()
        return (self._ok("Shutting down") if ok
                else self._err("Shutdown blocked. Set ALLOW_SHUTDOWN=true to enable."))

    def _restart(self, *_):
        self._say("Restarting the computer")
        time.sleep(1)
        ok = self.system.restart()
        return (self._ok("Restarting") if ok
                else self._err("Restart blocked. Set ALLOW_SHUTDOWN=true to enable."))

    def _lock_screen(self, *_):
        self.system.lock_screen()
        return self._ok("Screen locked")

    def _screenshot(self, value, _):
        path = self.system.screenshot(value or None)
        return (self._ok(f"Screenshot saved: {path}") if path
                else self._err("Screenshot failed"))

    def _new_tab(self, *_):
        self.browser.new_tab()
        return self._ok("New tab opened")

    def _close_tab(self, *_):
        self.browser.close_tab()
        return self._ok("Tab closed")

    def _next_tab(self, *_):
        self.browser.next_tab()
        return self._ok("Next tab")

    def _prev_tab(self, *_):
        self.browser.prev_tab()
        return self._ok("Previous tab")

    def _go_back(self, *_):
        self.browser.go_back()
        return self._ok("Going back")

    def _go_forward(self, *_):
        self.browser.go_forward()
        return self._ok("Going forward")

    def _refresh(self, *_):
        self.browser.refresh()
        return self._ok("Refreshing")

    def _scroll_up(self, value, _):
        amount = int(value) if value.isdigit() else None
        self.browser.scroll_up(amount)
        return self._ok("Scrolling up")

    def _scroll_down(self, value, _):
        amount = int(value) if value.isdigit() else None
        self.browser.scroll_down(amount)
        return self._ok("Scrolling down")

    def _search_google(self, value, _):
        self.browser.search_google(value)
        return self._ok(f"Searching Google for: {value}")

    def _search_youtube(self, value, _):
        self.browser.search_youtube(value)
        return self._ok(f"Searching YouTube for: {value}")

    def _open_url(self, value, _):
        self.browser.open_url(value)
        return self._ok(f"Opening {value}")

    def _minimize_window(self, *_):
        self.browser.minimize_window()
        return self._ok("Window minimized")

    def _maximize_window(self, *_):
        self.browser.maximize_window()
        return self._ok("Window maximized")

    def _close_window(self, *_):
        self.browser.close_window()
        return self._ok("Window closed")

    def _switch_window(self, *_):
        self.browser.switch_window()
        return self._ok("Switching window")

    def _open_folder(self, value, _):
        ok = self.files.open_folder(value)
        return self._ok(f"Opening {value} folder") if ok else self._err(f"Could not open {value}")

    def _create_file(self, value, _):
        ok = self.files.create_file(value)
        return self._ok(f"Created file: {value}") if ok else self._err(f"Could not create {value}")

    def _delete_file(self, value, _):
        ok = self.files.delete_file(value)
        return self._ok(f"Deleted: {value}") if ok else self._err(f"Could not delete {value}")

    def _search_file(self, value, _):
        results = self.files.search_file(value)
        if results:
            preview = results[:3]
            return self._ok(f"Found {len(results)} file(s): " + ", ".join(preview))
        return self._err(f"No files found matching '{value}'")

    def _press_key(self, value, _):
        ok = self.kb.press_key(value)
        return self._ok(f"Pressed: {value}") if ok else self._err(f"Key press failed: {value}")

    def _type_text(self, value, _):
        ok = self.kb.type_text(value)
        return self._ok(f"Typed text") if ok else self._err("Type failed")

    def _run_code(self, *_):
        self.vscode.run_code()
        return self._ok("Running code")

    def _write_code(self, value, _):
        # value is the code string produced by the script generator
        ok = self.vscode.write_code(value)
        return self._ok("Code written to editor") if ok else self._err("Write failed")

    def _debug_code(self, *_):
        self.vscode.start_debug()
        return self._ok("Debugging started")

    def _open_file_vscode(self, value, _):
        ok = self.vscode.open_file(value)
        return self._ok(f"Opened {value} in VS Code") if ok else self._err("Could not open file")

    def _open_terminal(self, *_):
        self.vscode.open_terminal()
        return self._ok("Terminal opened")

    def _run_terminal_command(self, value, _):
        ok = self.vscode.run_terminal_command(value)
        return self._ok(f"Command: {value}") if ok else self._err("Terminal command failed")

    def _generate_script(self, value, _):
        return self._ok(f"Script generation triggered for: {value}")

    def _improve_script(self, value, _):
        return self._ok(f"Script improvement triggered for: {value}")

    def _describe_screen(self, *_):
        return self._ok("Screen description triggered")

    def _show_help(self, *_):
        lines = [
            "Available voice commands:",
            "  open/close/focus <app>",
            "  volume up / volume down / mute",
            "  brightness up / brightness down",
            "  new tab / close tab / next tab / previous tab",
            "  scroll up / scroll down",
            "  search google for <query>",
            "  search youtube for <query>",
            "  open url <url>",
            "  open/create/delete/search file",
            "  open folder <name>",
            "  run code / debug / write code",
            "  open terminal",
            "  what is on screen",
            "  generate script <description>",
            "  shutdown / restart / lock screen / screenshot",
            "  repeat / stop",
        ]
        text = "\n".join(lines)
        self._say("Here are some things I can do.")
        return self._ok(text)

    def _repeat_last(self, *_):
        entry = self.history.last_successful()
        if entry:
            return self.route(entry.intent, entry.value, entry.raw_text)
        return self._err("Nothing to repeat.")

    def _stop(self, *_):
        return self._ok("Going to sleep. Say my name to wake me.")

    def _unknown(self, value, raw):
        return self._err(f"I didn't understand: '{raw}'")

    # ── Dispatch table ────────────────────────────────────────────────────────

    _DISPATCH: dict[str, callable] = {
        "open_app":            _open_app,
        "close_app":           _close_app,
        "focus_app":           _focus_app,
        "volume_up":           _volume_up,
        "volume_down":         _volume_down,
        "mute":                _mute,
        "brightness_up":       _brightness_up,
        "brightness_down":     _brightness_down,
        "shutdown":            _shutdown,
        "restart":             _restart,
        "lock_screen":         _lock_screen,
        "screenshot":          _screenshot,
        "new_tab":             _new_tab,
        "close_tab":           _close_tab,
        "next_tab":            _next_tab,
        "prev_tab":            _prev_tab,
        "go_back":             _go_back,
        "go_forward":          _go_forward,
        "refresh":             _refresh,
        "scroll_up":           _scroll_up,
        "scroll_down":         _scroll_down,
        "search_google":       _search_google,
        "search_youtube":      _search_youtube,
        "open_url":            _open_url,
        "minimize_window":     _minimize_window,
        "maximize_window":     _maximize_window,
        "close_window":        _close_window,
        "switch_window":       _switch_window,
        "open_folder":         _open_folder,
        "create_file":         _create_file,
        "delete_file":         _delete_file,
        "search_file":         _search_file,
        "press_key":           _press_key,
        "type_text":           _type_text,
        "run_code":            _run_code,
        "write_code":          _write_code,
        "debug_code":          _debug_code,
        "open_file_vscode":    _open_file_vscode,
        "open_terminal":       _open_terminal,
        "run_terminal_command":_run_terminal_command,
        "generate_script":     _generate_script,
        "improve_script":      _improve_script,
        "what_on_screen":      _describe_screen,
        "describe_screen":     _describe_screen,
        "show_help":           _show_help,
        "repeat_last":         _repeat_last,
        "stop":                _stop,
        "unknown":             _unknown,
    }
