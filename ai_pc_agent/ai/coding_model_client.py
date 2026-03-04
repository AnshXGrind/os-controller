"""ai_pc_agent/ai/coding_model_client.py

Coding-focused LLM client.  Points at a coding-optimised model (can be the
same as the default model if only one is available).
"""

from __future__ import annotations
from ai_pc_agent.ai.ollama_client import OllamaClient
from ai_pc_agent.utils import config
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.coding")

_CODE_SYSTEM = """\
You are an expert Python programmer and automation engineer.
Generate clean, runnable Python code.
Return ONLY the code — no explanations, no markdown fences unless specifically asked.
"""

_DEBUG_SYSTEM = """\
You are a Python debugger. Analyse the code and error, then:
1. Explain the bug briefly.
2. Return the FIXED code only (no markdown, no commentary outside code).
"""

_IMPROVE_SYSTEM = """\
You are a Python code optimisation expert.
Improve the given automation script for clarity, robustness and efficiency.
Return ONLY the improved code.
"""


class CodingModelClient:
    """Wraps OllamaClient with coding-specific prompts."""

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient(
            model=config.get("OLLAMA_CODING_MODEL")
        )

    def generate_script(self, description: str) -> str:
        """Generate a Python script from a natural language description."""
        prompt = (
            f"Write a complete, runnable Python script that does the following:\n"
            f"{description}\n\n"
            "Output pure Python code only."
        )
        return self.client.ask(prompt, system=_CODE_SYSTEM)

    def debug_code(self, code: str, error: str) -> str:
        """Return fixed code given original code and error message."""
        prompt = (
            f"Code with bug:\n```python\n{code}\n```\n\n"
            f"Error:\n{error}\n\n"
            "Return ONLY the fixed Python code."
        )
        return self.client.ask(prompt, system=_DEBUG_SYSTEM)

    def improve_script(self, code: str, goal: str = "") -> str:
        """Return an improved version of the given script."""
        note = f"\nImprovement goal: {goal}" if goal else ""
        prompt = f"Script to improve:\n```python\n{code}\n```{note}\n\nReturn improved code only."
        return self.client.ask(prompt, system=_IMPROVE_SYSTEM)

    def explain_code(self, code: str) -> str:
        """Return a plain-English explanation of the code."""
        prompt = f"Explain what this Python code does:\n```python\n{code}\n```"
        return self.client.ask(prompt)
