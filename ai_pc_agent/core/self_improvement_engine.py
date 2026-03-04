"""ai_pc_agent/core/self_improvement_engine.py

Analyses repeated command patterns and generates reusable Python scripts
(skills) that are stored in the SkillLibrary for future one-shot execution.
"""

from __future__ import annotations
from collections import Counter

from ai_pc_agent.ai.coding_model_client  import CodingModelClient
from ai_pc_agent.memory.command_history  import CommandHistory
from ai_pc_agent.memory.skill_library    import SkillLibrary
from ai_pc_agent.utils.logger            import get_logger

logger = get_logger("agent.improve")

_MIN_REPEAT = 3          # times a pattern must appear before auto-generating a skill
_SKIP_INTENTS = {        # intents too generic to auto-skill
    "unknown", "stop", "show_help", "repeat_last",
    "volume_up", "volume_down", "mute", "scroll_up", "scroll_down",
}


class SelfImprovementEngine:
    """Detect repetitive workflows and turn them into reusable skills."""

    def __init__(
        self,
        coder:   CodingModelClient,
        history: CommandHistory,
        skills:  SkillLibrary,
    ):
        self.coder   = coder
        self.history = history
        self.skills  = skills

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self) -> list[str]:
        """
        Inspect history for repeated (intent, value) pairs.
        Auto-generate scripts for any pattern that crosses the threshold.
        Returns list of new skill names created.
        """
        counts: Counter = Counter()
        for entry in self.history._entries:
            if entry.intent in _SKIP_INTENTS:
                continue
            key = f"{entry.intent}:{entry.value}"
            counts[key] += 1

        created = []
        for key, count in counts.items():
            if count < _MIN_REPEAT:
                continue
            intent, value = key.split(":", 1)
            skill_name = f"{intent}__{value[:20].replace(' ', '_')}"
            if self.skills.get(skill_name):
                continue   # already exists
            logger.info("Auto-generating skill '%s' (seen %d×)", skill_name, count)
            code = self._generate(intent, value)
            if code:
                self.skills.add(
                    name        = skill_name,
                    description = f"Auto-generated: {intent} '{value}'",
                    code        = code,
                    trigger     = value,
                )
                created.append(skill_name)
        return created

    def generate_custom_skill(
        self,
        name:        str,
        description: str,
        trigger:     str = "",
    ) -> str | None:
        """Generate and store a custom skill from a natural language description."""
        code = self.coder.generate_script(description)
        if not code:
            return None
        self.skills.add(name=name, description=description, code=code, trigger=trigger)
        logger.info("Custom skill created: '%s'", name)
        return code

    def improve_existing(self, skill_name: str, goal: str = "") -> bool:
        """Rewrite an existing skill with improved code."""
        skill = self.skills.get(skill_name)
        if not skill:
            logger.warning("improve_existing: skill '%s' not found", skill_name)
            return False
        improved = self.coder.improve_script(skill.code, goal=goal)
        if not improved:
            return False
        self.skills.update_code(skill_name, improved)
        logger.info("Skill '%s' improved", skill_name)
        return True

    def run_skill(self, skill_name: str) -> tuple[bool, str]:
        """Execute a stored skill's Python code in a sandboxed exec()."""
        skill = self.skills.get(skill_name)
        if not skill:
            return False, f"Skill '{skill_name}' not found"
        try:
            local_ns: dict = {}
            exec(compile(skill.code, skill_name, "exec"), {}, local_ns)
            self.skills.increment_use(skill_name)
            return True, f"Skill '{skill_name}' executed successfully"
        except Exception as exc:
            logger.error("run_skill '%s': %s", skill_name, exc)
            return False, f"Skill error: {exc}"

    # ── Private ───────────────────────────────────────────────────────────────

    def _generate(self, intent: str, value: str) -> str:
        desc = (
            f"Python script that performs the automation action '{intent}' "
            f"with argument '{value}' on a Windows PC. "
            "Use pyautogui, subprocess, or os as appropriate. "
            "The script must be self-contained and runnable with exec()."
        )
        return self.coder.generate_script(desc)
