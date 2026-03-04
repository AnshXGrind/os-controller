"""ai_pc_agent/ai/llm_reasoning.py

Higher-level reasoning helpers built on top of OllamaClient.
"""

from __future__ import annotations
from typing import Any

from ai_pc_agent.ai.ollama_client import OllamaClient
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.reasoning")

_INTENT_SYSTEM = """\
You are an intent classifier for a computer control AI assistant.
The user will say a voice command. Your job is to return ONLY a compact JSON object
with the fields below — no prose, no markdown fences, nothing else.

JSON schema:
{
  "intent": "<one of the allowed intents>",
  "value":  "<argument string or empty string>",
  "steps":  ["<step1>", "<step2>"]
}

Allowed intents:
open_app | close_app | focus_app |
volume_up | volume_down | mute |
brightness_up | brightness_down |
shutdown | restart | lock_screen | screenshot |
new_tab | close_tab | next_tab | prev_tab |
go_back | go_forward | refresh |
scroll_up | scroll_down |
search_google | search_youtube | open_url |
minimize_window | maximize_window | close_window | switch_window |
open_folder | create_file | delete_file | search_file |
press_key | type_text | move_mouse | click_mouse |
run_code | write_code | debug_code | open_file_vscode |
open_terminal | run_terminal_command |
generate_script | improve_script |
what_on_screen | describe_screen |
remember_context | recall_context |
show_help | repeat_last | stop | unknown

Rules:
- Use "value" for the main argument (app name, query, key, URL, filename, etc.)
- Use "steps" ONLY if the task needs multiple sequential actions (otherwise [])
- Always choose the most specific intent
- Return JSON only
"""

_PLAN_SYSTEM = """\
You are a task planner for a computer control AI assistant.
Given a complex user command, break it into an ordered list of simple steps.
Return ONLY a JSON array of step strings, e.g.:
["open browser", "go to youtube.com", "search for python tutorials", "press enter"]
No prose, no markdown.
"""

_SCREEN_SYSTEM = """\
You are a screen analysis assistant. The user will describe or ask about what is visible
on their computer screen. Answer concisely and practically. If asked to identify an
application, state only the app name. If unsure, say so.
"""


class LLMReasoning:
    """Higher-level reasoning API."""

    def __init__(self, client: OllamaClient):
        self.client = client

    # ── Intent interpretation ─────────────────────────────────────────────────

    def interpret_command(self, command: str) -> dict:
        """Return structured intent dict from a raw voice command."""
        result = self.client.ask_json(
            prompt=f'User command: "{command}"',
            system=_INTENT_SYSTEM,
        )
        if result and "intent" in result:
            result.setdefault("value", "")
            result.setdefault("steps", [])
            return result
        return {"intent": "unknown", "value": command, "steps": []}

    # ── Task planning ─────────────────────────────────────────────────────────

    def plan_task(self, command: str) -> list[str]:
        """Break a complex command into ordered steps."""
        import json, re
        raw = self.client.ask(
            prompt=f'User command: "{command}"',
            system=_PLAN_SYSTEM,
        )
        raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
        try:
            steps = json.loads(raw)
            if isinstance(steps, list):
                return [str(s) for s in steps]
        except Exception:
            pass
        # Fallback: split by numbered lines
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        steps = []
        for line in lines:
            line = re.sub(r"^[\d]+[.)]\s*", "", line)
            if line:
                steps.append(line)
        return steps or [command]

    # ── Screen understanding ──────────────────────────────────────────────────

    def analyse_screen(self, description: str, question: str = "") -> str:
        """Use LLM to reason about a screen description."""
        if not question:
            question = "What is the user currently doing on this computer?"
        prompt = f"Screen description:\n{description}\n\nQuestion: {question}"
        return self.client.ask(prompt, system=_SCREEN_SYSTEM)

    # ── General Q&A ──────────────────────────────────────────────────────────

    def ask(self, prompt: str, system: str = "") -> str:
        return self.client.ask(prompt, system=system)
