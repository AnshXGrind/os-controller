"""ai_pc_agent/vision/screen_understanding.py

Combine screen capture with LLM analysis to understand what is on screen.
"""

from __future__ import annotations
from ai_pc_agent.vision.screen_capture import ScreenCapture
from ai_pc_agent.ai.llm_reasoning import LLMReasoning
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.vision")


class ScreenUnderstanding:
    """Answer questions about the current screen using LLM reasoning."""

    def __init__(self, capture: ScreenCapture, reasoning: LLMReasoning):
        self.capture   = capture
        self.reasoning = reasoning

    # ── Public API ────────────────────────────────────────────────────────────

    def describe(self) -> str:
        """Return a short natural-language description of the current screen."""
        context = self.capture.get_screen_text()
        return self.reasoning.analyse_screen(
            context,
            question="Briefly describe what is visible on the screen right now.",
        )

    def what_app(self) -> str:
        """Return the name of the currently active application."""
        context = self.capture.get_screen_text()
        return self.reasoning.analyse_screen(
            context,
            question="What application is currently in focus? Reply with its name only.",
        )

    def answer(self, question: str) -> str:
        """Answer an arbitrary question about the current screen."""
        context = self.capture.get_screen_text()
        return self.reasoning.analyse_screen(context, question=question)

    def capture_and_describe(self, save: bool = False) -> str:
        """Optionally save a screenshot then describe it."""
        if save:
            path = self.capture.capture_and_save()
            logger.info("Screenshot saved: %s", path)
        return self.describe()
