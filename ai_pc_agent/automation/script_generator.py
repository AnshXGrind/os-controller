"""ai_pc_agent/automation/script_generator.py

Generate, validate, save, and run Python automation scripts via the coding LLM.
"""

from __future__ import annotations
import ast
import subprocess
import sys
import time
from pathlib import Path

from ai_pc_agent.ai.coding_model_client import CodingModelClient
from ai_pc_agent.utils.logger           import get_logger

logger = get_logger("agent.scriptgen")

_SCRIPTS_DIR = Path("generated_scripts")


class ScriptGenerator:
    """Request Python automation scripts from the LLM and manage them."""

    def __init__(self, coder: CodingModelClient, scripts_dir: str | None = None):
        self.coder       = coder
        self.scripts_dir = Path(scripts_dir or _SCRIPTS_DIR)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate ──────────────────────────────────────────────────────────────

    def generate(self, description: str, filename: str | None = None) -> tuple[str, Path | None]:
        """
        Generate a Python script from *description*.
        Validates syntax; saves to disk.
        Returns (code_string, saved_path | None).
        """
        logger.info("Generating script: %s", description[:80])
        code = self.coder.generate_script(description)
        if not code:
            logger.error("Code generation returned empty string.")
            return "", None

        # Syntax validation
        try:
            ast.parse(code)
        except SyntaxError as exc:
            logger.warning("Generated code has syntax error (%s). Attempting auto-fix.", exc)
            code = self._fix_syntax(code, str(exc))
            if not code:
                return "", None

        # Save
        fname = filename or f"script_{int(time.time())}.py"
        path  = self.scripts_dir / fname
        path.write_text(code, encoding="utf-8")
        logger.info("Script saved: %s (%d lines)", path, len(code.splitlines()))
        return code, path

    # ── Execute ───────────────────────────────────────────────────────────────

    def run(self, path: str | Path, timeout: int = 30) -> tuple[bool, str]:
        """Execute a saved script in a subprocess. Returns (success, output)."""
        p = Path(path)
        if not p.exists():
            return False, f"Script not found: {path}"
        try:
            result = subprocess.run(
                [sys.executable, str(p)],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                logger.info("Script ran successfully: %s", p.name)
                return True, result.stdout.strip()
            else:
                logger.warning("Script error: %s", result.stderr[:200])
                return False, result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, f"Script timed out after {timeout}s"
        except Exception as exc:
            return False, str(exc)

    def run_and_fix(
        self,
        path:      str | Path,
        max_fixes: int = 2,
        timeout:   int = 30,
    ) -> tuple[bool, str]:
        """Run a script; if it fails, ask the LLM to fix it (up to max_fixes times)."""
        p = Path(path)
        for attempt in range(max_fixes + 1):
            ok, output = self.run(p, timeout=timeout)
            if ok:
                return True, output
            if attempt < max_fixes:
                logger.info("Auto-fixing script (attempt %d/%d)…", attempt + 1, max_fixes)
                code = p.read_text(encoding="utf-8")
                fixed = self.coder.debug_code(code, output)
                if fixed:
                    try:
                        ast.parse(fixed)
                        p.write_text(fixed, encoding="utf-8")
                    except SyntaxError:
                        logger.warning("Fixed code has syntax error — skipping.")
                        break
        return False, output

    # ── List ─────────────────────────────────────────────────────────────────

    def list_scripts(self) -> list[Path]:
        return sorted(self.scripts_dir.glob("*.py"))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fix_syntax(self, code: str, error: str) -> str:
        """Ask the LLM to fix a syntax error in the generated code."""
        fixed = self.coder.debug_code(code, f"SyntaxError: {error}")
        try:
            ast.parse(fixed)
            return fixed
        except Exception:
            return ""
