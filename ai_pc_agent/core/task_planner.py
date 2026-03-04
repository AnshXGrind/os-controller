"""ai_pc_agent/core/task_planner.py

Break complex multi-step commands into an ordered plan of simple actions.
"""

from __future__ import annotations
import re

from ai_pc_agent.ai.llm_reasoning    import LLMReasoning
from ai_pc_agent.core.intent_interpreter import IntentInterpreter
from ai_pc_agent.utils.logger        import get_logger

logger = get_logger("agent.planner")

# ── Simple heuristics for multi-step detection ───────────────────────────────

_MULTI_STEP_SIGNALS = [
    "and then", "after that", "then", " and ", "first", "next",
    "download", "summarize", "summarise", "search and", "open and",
]

def _looks_multi_step(command: str) -> bool:
    cl = command.lower()
    return any(sig in cl for sig in _MULTI_STEP_SIGNALS)


class TaskPlanner:
    """Break a command into steps and return a list of (intent, value) pairs."""

    def __init__(
        self,
        reasoning:    LLMReasoning | None = None,
        interpreter:  IntentInterpreter | None = None,
        use_llm:      bool = True,
    ):
        self.reasoning   = reasoning
        self.interpreter = interpreter or IntentInterpreter(
            reasoning=reasoning, use_llm=use_llm
        )
        self.use_llm     = use_llm

    # ── Public API ────────────────────────────────────────────────────────────

    def plan(self, command: str) -> list[dict]:
        """
        Return a list of step dicts: [{'intent': ..., 'value': ...}, ...]
        For simple commands, returns a single-item list.
        """
        # If the first LLM interpret already returned steps, use those
        interpreted = self.interpreter.interpret(command)
        if interpreted.get("steps"):
            steps = interpreted["steps"]
            logger.info("Planner: %d LLM-provided step(s)", len(steps))
            return [self.interpreter.interpret(s) for s in steps]

        # Heuristic: looks complex → ask LLM to plan
        if self.use_llm and self.reasoning and _looks_multi_step(command):
            steps = self._llm_plan(command)
            if len(steps) > 1:
                logger.info("Planner: %d LLM-planned step(s)", len(steps))
                return [self.interpreter.interpret(s) for s in steps]

        # Simple single step
        return [interpreted]

    def _llm_plan(self, command: str) -> list[str]:
        try:
            return self.reasoning.plan_task(command)
        except Exception as exc:
            logger.warning("task_plan failed: %s", exc)
            return [command]

    def describe_plan(self, steps: list[dict]) -> str:
        lines = [f"{i+1}. [{s['intent']}] {s['value']}" for i, s in enumerate(steps)]
        return "\n".join(lines)
