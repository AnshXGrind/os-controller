"""ai_pc_agent/core/self_healing_engine.py

Detect runtime errors, analyse stack traces, and generate fixes
automatically using the coding LLM.
"""

from __future__ import annotations
import traceback
from pathlib import Path

from ai_pc_agent.ai.coding_model_client import CodingModelClient
from ai_pc_agent.utils.logger           import get_logger

logger = get_logger("agent.heal")


class SelfHealingEngine:
    """
    Auto-repair engine.
    Given a traceback + optional source code, asks the coding LLM to:
      1. Diagnose the bug
      2. Return a fixed version
    """

    def __init__(self, coder: CodingModelClient):
        self.coder  = coder
        self._fixes: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def heal(
        self,
        error_msg:   str,
        tb:          str = "",
        source_file: str = "",
    ) -> str:
        """
        Attempt to generate a fix.
        Returns the fix suggestion string (code or diff), or "".
        """
        code = ""
        if source_file:
            try:
                code = Path(source_file).read_text(encoding="utf-8")
            except Exception:
                pass

        logger.info("Self-healing: %s", error_msg[:120])
        suggestion = self.coder.debug_code(
            code  = code or "(source unavailable)",
            error = f"{error_msg}\n\nTraceback:\n{tb}",
        )
        if suggestion:
            self._fixes.append({
                "error": error_msg[:200],
                "fix":   suggestion[:500],
                "file":  source_file,
            })
            logger.info("Heal suggestion generated (%d chars)", len(suggestion))
        return suggestion

    def heal_exception(self, exc: Exception, source_file: str = "") -> str:
        """Convenience wrapper that takes a live exception."""
        tb = traceback.format_exc()
        return self.heal(str(exc), tb=tb, source_file=source_file)

    def apply_fix(self, source_file: str, fixed_code: str) -> bool:
        """Overwrite *source_file* with *fixed_code* after syntax check."""
        import ast
        try:
            ast.parse(fixed_code)
        except SyntaxError as exc:
            logger.error("apply_fix: syntax error in fix — not applied: %s", exc)
            return False
        try:
            p = Path(source_file)
            # Backup
            bak = p.with_suffix(".py.bak")
            bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            p.write_text(fixed_code, encoding="utf-8")
            logger.info("Fix applied to '%s' (backup: %s)", source_file, bak.name)
            return True
        except Exception as exc:
            logger.error("apply_fix error: %s", exc)
            return False

    def recent_fixes(self, n: int = 5) -> list[dict]:
        return self._fixes[-n:]
