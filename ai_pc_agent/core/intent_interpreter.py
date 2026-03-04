"""ai_pc_agent/core/intent_interpreter.py

Turn a raw voice command into a structured intent dict using the LLM.
Keyword fallback when Ollama is unavailable.
"""

from __future__ import annotations
import re

from ai_pc_agent.ai.llm_reasoning import LLMReasoning
from ai_pc_agent.utils.logger     import get_logger

logger = get_logger("agent.intent")

# ── Keyword fallback map ──────────────────────────────────────────────────────

_KW: list[tuple[list[str], str]] = [
    (["open", "launch", "start"],                            "open_app"),
    (["close", "quit", "exit", "kill"],                      "close_app"),
    (["volume up", "louder", "increase volume"],             "volume_up"),
    (["volume down", "quieter", "decrease volume"],          "volume_down"),
    (["mute", "silence"],                                    "mute"),
    (["brightness up", "brighter"],                          "brightness_up"),
    (["brightness down", "dimmer"],                          "brightness_down"),
    (["shutdown", "shut down", "power off"],                 "shutdown"),
    (["restart", "reboot"],                                  "restart"),
    (["lock", "lock screen", "lock computer"],               "lock_screen"),
    (["screenshot", "take screenshot"],                      "screenshot"),
    (["new tab"],                                            "new_tab"),
    (["close tab"],                                          "close_tab"),
    (["next tab"],                                           "next_tab"),
    (["previous tab", "prev tab", "back tab"],               "prev_tab"),
    (["go back", "back"],                                    "go_back"),
    (["go forward", "forward"],                              "go_forward"),
    (["refresh", "reload"],                                  "refresh"),
    (["scroll up"],                                          "scroll_up"),
    (["scroll down"],                                        "scroll_down"),
    (["search google", "google"],                            "search_google"),
    (["search youtube", "youtube"],                          "search_youtube"),
    (["open url", "go to", "navigate to"],                   "open_url"),
    (["minimize"],                                           "minimize_window"),
    (["maximize"],                                           "maximize_window"),
    (["close window"],                                       "close_window"),
    (["switch window", "alt tab"],                           "switch_window"),
    (["open folder"],                                        "open_folder"),
    (["create file", "make file", "new file"],               "create_file"),
    (["delete file", "remove file"],                         "delete_file"),
    (["search file", "find file"],                           "search_file"),
    (["press key", "press"],                                 "press_key"),
    (["type", "type text", "write"],                         "type_text"),
    (["run code", "execute", "run script"],                  "run_code"),
    (["write code", "generate code"],                        "write_code"),
    (["debug", "fix code"],                                  "debug_code"),
    (["open terminal", "open console"],                      "open_terminal"),
    (["generate script", "create script"],                   "generate_script"),
    (["improve script", "optimise script"],                  "improve_script"),
    (["what on screen", "describe screen", "what is visible"], "describe_screen"),
    (["help"],                                               "show_help"),
    (["repeat", "again", "redo"],                            "repeat_last"),
    (["stop", "sleep", "pause", "stand by"],                 "stop"),
]


def _kw_parse(text: str) -> dict:
    tl = text.lower()
    for phrases, intent in _KW:
        for phrase in phrases:
            if phrase in tl:
                # Extract value: everything after the triggering phrase
                idx = tl.find(phrase)
                value = text[idx + len(phrase):].strip()
                # Also check "for X", "with X" patterns
                m = re.search(r"\s+(?:for|with|of|to)\s+(.+)$", value, re.I)
                if m:
                    value = m.group(1).strip()
                return {"intent": intent, "value": value, "steps": []}
    return {"intent": "unknown", "value": text, "steps": []}


class IntentInterpreter:
    """Convert a raw command string into a structured intent dict."""

    def __init__(
        self,
        reasoning: LLMReasoning | None = None,
        use_llm: bool = True,
        fallback_kw: bool = True,
    ):
        self.reasoning   = reasoning
        self.use_llm     = use_llm and reasoning is not None
        self.fallback_kw = fallback_kw
        self._llm_ok: bool | None = None

    def interpret(self, command: str) -> dict:
        """Return {'intent': str, 'value': str, 'steps': [str]}."""
        if self.use_llm and self._llm_available():
            result = self._try_llm(command)
            if result:
                return result
        if self.fallback_kw:
            return _kw_parse(command)
        return {"intent": "unknown", "value": command, "steps": []}

    def _try_llm(self, command: str) -> dict | None:
        try:
            result = self.reasoning.interpret_command(command)
            if result.get("intent", "unknown") != "unknown":
                return result
        except Exception as exc:
            logger.warning("LLM interpret failed: %s", exc)
            self._llm_ok = False
        return None

    def _llm_available(self) -> bool:
        if self._llm_ok is None:
            self._llm_ok = self.reasoning.client.is_available() if self.reasoning else False
        return self._llm_ok

    def reset_llm(self):
        self._llm_ok = None
        if self.reasoning:
            self.reasoning.client.reset_availability()
