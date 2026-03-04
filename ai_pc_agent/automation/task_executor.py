"""ai_pc_agent/automation/task_executor.py

Execute individual tasks with retry logic, timeout protection,
and structured result reporting.
"""

from __future__ import annotations
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Any

from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.executor")


@dataclass
class TaskResult:
    task_id:    str
    intent:     str
    value:      str
    success:    bool
    response:   str
    elapsed_ms: float
    error:      str = ""
    retries:    int = 0


class TaskExecutor:
    """
    Execute single tasks with:
    - configurable retry logic
    - per-task timeout guard
    - structured result objects
    - per-task pre/post hooks
    """

    def __init__(
        self,
        default_retries:  int   = 2,
        retry_delay:      float = 0.5,
        action_cooldown:  float = 0.4,
    ):
        self.default_retries  = default_retries
        self.retry_delay      = retry_delay
        self.action_cooldown  = action_cooldown
        self._pre_hooks:  list[Callable] = []
        self._post_hooks: list[Callable] = []

    # ── Hooks ─────────────────────────────────────────────────────────────────

    def add_pre_hook(self, fn: Callable):
        self._pre_hooks.append(fn)

    def add_post_hook(self, fn: Callable):
        self._post_hooks.append(fn)

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(
        self,
        task_id:  str,
        intent:   str,
        value:    str,
        handler:  Callable[[], tuple[bool, str]],
        retries:  int | None = None,
    ) -> TaskResult:
        """
        Execute *handler* with retry logic.
        handler must be a zero-arg callable returning (success, response_text).
        """
        max_tries = (retries if retries is not None else self.default_retries) + 1
        last_error = ""
        attempt    = 0

        # Pre-hooks
        for hook in self._pre_hooks:
            try:
                hook(task_id, intent, value)
            except Exception:
                pass

        t0 = time.perf_counter()

        for attempt in range(max_tries):
            try:
                success, response = handler()
                elapsed = (time.perf_counter() - t0) * 1000
                result = TaskResult(
                    task_id    = task_id,
                    intent     = intent,
                    value      = value,
                    success    = success,
                    response   = response,
                    elapsed_ms = elapsed,
                    retries    = attempt,
                )
                if success or attempt == max_tries - 1:
                    break
                logger.debug("Task '%s' returned failure on attempt %d, retrying…", task_id, attempt + 1)
                time.sleep(self.retry_delay)
            except Exception as exc:
                last_error = traceback.format_exc()
                logger.warning("Task '%s' attempt %d exception: %s", task_id, attempt + 1, exc)
                if attempt < max_tries - 1:
                    time.sleep(self.retry_delay)
                else:
                    elapsed = (time.perf_counter() - t0) * 1000
                    result = TaskResult(
                        task_id    = task_id,
                        intent     = intent,
                        value      = value,
                        success    = False,
                        response   = f"Error: {exc}",
                        elapsed_ms = elapsed,
                        error      = last_error,
                        retries    = attempt,
                    )

        # Post-hooks
        for hook in self._post_hooks:
            try:
                hook(result)
            except Exception:
                pass

        time.sleep(self.action_cooldown)
        return result

    def execute_many(
        self,
        tasks: list[dict],
        handler_fn: Callable[[str, str], tuple[bool, str]],
    ) -> list[TaskResult]:
        """
        Execute a list of task dicts [{intent, value}] sequentially.
        handler_fn(intent, value) → (success, response)
        """
        results = []
        for i, task in enumerate(tasks):
            intent = task.get("intent", "unknown")
            value  = task.get("value",  "")
            result = self.execute(
                task_id = f"step_{i}",
                intent  = intent,
                value   = value,
                handler = lambda i=intent, v=value: handler_fn(i, v),
            )
            results.append(result)
            if not result.success and task.get("stop_on_failure", False):
                logger.info("Stopping task chain at step %d (stop_on_failure)", i)
                break
        return results
