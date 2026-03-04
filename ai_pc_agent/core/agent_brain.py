"""ai_pc_agent/core/agent_brain.py

Central brain: wires together the interpreter, planner, router, memory,
vision, and self-improvement engine.
"""

from __future__ import annotations
import time

from ai_pc_agent.ai.ollama_client           import OllamaClient
from ai_pc_agent.ai.llm_reasoning           import LLMReasoning
from ai_pc_agent.ai.coding_model_client     import CodingModelClient
from ai_pc_agent.core.intent_interpreter    import IntentInterpreter
from ai_pc_agent.core.task_planner          import TaskPlanner
from ai_pc_agent.core.command_router        import CommandRouter
from ai_pc_agent.core.self_improvement_engine import SelfImprovementEngine
from ai_pc_agent.memory.context_memory      import ContextMemory
from ai_pc_agent.memory.command_history     import CommandHistory
from ai_pc_agent.memory.skill_library       import SkillLibrary
from ai_pc_agent.vision.screen_understanding import ScreenUnderstanding
from ai_pc_agent.utils import config
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.brain")


class AgentBrain:
    """
    Orchestrates a complete pipeline:
        raw_text
          → check learned skills
          → intent interpret
          → task plan
          → execute steps
          → update memory
    """

    def __init__(
        self,
        router:      CommandRouter,
        interpreter: IntentInterpreter,
        planner:     TaskPlanner,
        history:     CommandHistory,
        context:     ContextMemory,
        skills:      SkillLibrary,
        improver:    SelfImprovementEngine,
        vision:      ScreenUnderstanding | None = None,
        tts=None,
    ):
        self.router      = router
        self.interpreter = interpreter
        self.planner     = planner
        self.history     = history
        self.context     = context
        self.skills      = skills
        self.improver    = improver
        self.vision      = vision
        self.tts         = tts

    # ── Main entry ────────────────────────────────────────────────────────────

    def process(self, raw: str) -> tuple[bool, str]:
        """
        Process a raw voice command end-to-end.
        Returns (overall_success, final_response_text).
        """
        self.context.add_user(raw)

        # 1. Check if a learned skill matches
        skill = self.skills.find_by_trigger(raw)
        if skill:
            logger.info("Matched skill: '%s'", skill.name)
            success, response = self.improver.run_skill(skill.name)
            self._record(raw, "skill:" + skill.name, skill.name, success)
            self.context.add_assistant(response)
            return success, response

        # 2. Handle screen-related commands early
        if any(kw in raw.lower() for kw in ("screen", "visible", "what is on", "describe")):
            if self.vision:
                desc = self.vision.capture_and_describe()
                self.context.add_assistant(desc)
                return True, desc

        # 3. Generate script commands
        if any(kw in raw.lower() for kw in ("generate script", "create script", "write a script")):
            return self._handle_script_generation(raw)

        # 4. Plan → execute
        steps = self.planner.plan(raw)
        logger.info("Plan: %d step(s)", len(steps))

        results: list[tuple[bool, str]] = []
        for step in steps:
            intent = step.get("intent", "unknown")
            value  = step.get("value", "")
            success, response = self.router.route(intent, value, raw=raw)
            self._record(raw, intent, value, success)
            results.append((success, response))
            if self.tts and response:
                self.tts.speak_async(response)
            time.sleep(float(config.get("ACTION_COOLDOWN", 0.5)))

        # Aggregate
        all_ok  = all(r[0] for r in results)
        summary = " | ".join(r[1] for r in results if r[1])
        self.context.add_assistant(summary)
        return all_ok, summary

    # ── Periodic maintenance ──────────────────────────────────────────────────

    def maybe_improve(self, every: int = 20):
        """Call occasionally to auto-generate skills from repeated commands."""
        if len(self.history) % every == 0 and len(self.history) > 0:
            new_skills = self.improver.analyse()
            if new_skills:
                logger.info("Self-improvement: created skills %s", new_skills)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _record(self, raw: str, intent: str, value: str, success: bool):
        self.history.add(raw_text=raw, intent=intent, value=value, success=success)

    def _handle_script_generation(self, raw: str) -> tuple[bool, str]:
        # Extract description after "script" keyword
        import re
        m = re.search(r"(?:generate|create|write)\s+(?:a\s+)?(?:python\s+)?script\s+(?:that\s+|to\s+|for\s+)?(.+)", raw, re.I)
        description = m.group(1).strip() if m else raw
        code = self.improver.generate_custom_skill(
            name        = f"custom_{int(time.time())}",
            description = description,
            trigger     = description[:30],
        )
        if code:
            msg = f"Script generated ({len(code.splitlines())} lines). Saved to skill library."
            self.context.add_assistant(msg)
            return True, msg
        return False, "Script generation failed."
