"""ai_pc_agent/automation/workflow_engine.py

Define and execute reusable multi-step workflows.
A workflow is an ordered list of steps; each step is an (intent, value) pair.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from ai_pc_agent.automation.task_executor import TaskExecutor, TaskResult
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.workflow")


@dataclass
class WorkflowStep:
    intent:           str
    value:            str = ""
    delay_after:      float = 0.3
    stop_on_failure:  bool  = False


@dataclass
class Workflow:
    name:        str
    description: str
    steps:       list[WorkflowStep] = field(default_factory=list)
    trigger:     str = ""           # optional voice phrase that triggers this workflow

    def to_dict(self) -> dict:
        d = asdict(self); return d

    @staticmethod
    def from_dict(d: dict) -> "Workflow":
        steps = [WorkflowStep(**s) for s in d.pop("steps", [])]
        return Workflow(**d, steps=steps)


class WorkflowEngine:
    """Run, save, and load named multi-step workflows."""

    _BUILTIN: list[dict] = [
        {
            "name":        "youtube_search",
            "description": "Open browser and search YouTube",
            "trigger":     "search youtube",
            "steps": [
                {"intent": "open_app",       "value": "chrome",          "delay_after": 1.5},
                {"intent": "search_youtube", "value": "{query}",         "delay_after": 0.5},
            ],
        },
        {
            "name":        "google_search",
            "description": "Open browser and search Google",
            "trigger":     "search google",
            "steps": [
                {"intent": "open_app",      "value": "chrome",  "delay_after": 1.5},
                {"intent": "search_google", "value": "{query}", "delay_after": 0.5},
            ],
        },
        {
            "name":        "open_vscode_terminal",
            "description": "Open VS Code and launch terminal",
            "trigger":     "vscode terminal",
            "steps": [
                {"intent": "open_app",      "value": "vscode", "delay_after": 2.0},
                {"intent": "open_terminal", "value": "",       "delay_after": 0.5},
            ],
        },
    ]

    def __init__(
        self,
        executor:   TaskExecutor,
        router_fn:  Callable[[str, str], tuple[bool, str]],
        save_path:  str = "workflows.json",
    ):
        self.executor  = executor
        self.router_fn = router_fn
        self.save_path = Path(save_path)
        self._flows: dict[str, Workflow] = {}
        self._load_builtins()
        self._load_file()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, workflow: Workflow):
        self._flows[workflow.name.lower()] = workflow
        self._save()
        logger.info("Workflow registered: '%s'", workflow.name)

    def remove(self, name: str) -> bool:
        key = name.lower()
        if key in self._flows:
            del self._flows[key]
            self._save()
            return True
        return False

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Workflow | None:
        return self._flows.get(name.lower())

    def find_by_trigger(self, text: str) -> Workflow | None:
        tl = text.lower()
        for wf in self._flows.values():
            if wf.trigger and wf.trigger.lower() in tl:
                return wf
        return None

    def list_all(self) -> list[Workflow]:
        return list(self._flows.values())

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, name_or_workflow: str | Workflow, context: dict | None = None) -> list[TaskResult]:
        """Execute a workflow by name or object.  context replaces {placeholders}."""
        wf = (self.get(name_or_workflow) if isinstance(name_or_workflow, str)
              else name_or_workflow)
        if not wf:
            logger.warning("Workflow '%s' not found", name_or_workflow)
            return []

        ctx     = context or {}
        results = []
        logger.info("Running workflow '%s' (%d steps)", wf.name, len(wf.steps))

        for i, step in enumerate(wf.steps):
            intent = step.intent
            value  = step.value
            # Replace {placeholders}
            for k, v in ctx.items():
                value = value.replace(f"{{{k}}}", str(v))

            result = self.executor.execute(
                task_id = f"{wf.name}_step_{i}",
                intent  = intent,
                value   = value,
                handler = lambda i=intent, v=value: self.router_fn(i, v),
            )
            results.append(result)
            time.sleep(step.delay_after)

            if not result.success and step.stop_on_failure:
                logger.info("Workflow '%s' halted at step %d", wf.name, i)
                break

        ok = sum(1 for r in results if r.success)
        logger.info("Workflow '%s' done: %d/%d steps ok", wf.name, ok, len(results))
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_builtins(self):
        for d in self._BUILTIN:
            wf = Workflow.from_dict(dict(d))
            self._flows[wf.name.lower()] = wf

    def _load_file(self):
        if not self.save_path.exists():
            return
        try:
            data = json.loads(self.save_path.read_text(encoding="utf-8"))
            for item in data:
                wf = Workflow.from_dict(item)
                self._flows[wf.name.lower()] = wf
            logger.info("Loaded %d workflow(s) from %s", len(data), self.save_path)
        except Exception as exc:
            logger.warning("Workflow load failed: %s", exc)

    def _save(self):
        # Only persist non-builtin workflows
        builtin_names = {d["name"].lower() for d in self._BUILTIN}
        custom = [v.to_dict() for k, v in self._flows.items() if k not in builtin_names]
        try:
            self.save_path.write_text(json.dumps(custom, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Workflow save failed: %s", exc)
