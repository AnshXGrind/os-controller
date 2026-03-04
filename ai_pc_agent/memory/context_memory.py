"""ai_pc_agent/memory/context_memory.py

Short-term memory: current application, recent commands, conversation history.
"""

from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field

from ai_pc_agent.utils import config
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.context")


@dataclass
class ContextItem:
    role:    str   # "user" | "assistant" | "system"
    content: str
    ts:      float = field(default_factory=time.time)


class ContextMemory:
    """Sliding window of recent dialogue turns and environmental context."""

    def __init__(self, max_items: int | None = None):
        self.max_items    = max_items or int(config.get("MAX_CONTEXT_ITEMS", 20))
        self._items:      deque[ContextItem] = deque(maxlen=self.max_items)
        self.active_app:  str = "unknown"
        self.cwd:         str = ""
        self.screen_desc: str = ""

    # ── Dialogue ──────────────────────────────────────────────────────────────

    def add_user(self, content: str):
        self._items.append(ContextItem(role="user", content=content))

    def add_assistant(self, content: str):
        self._items.append(ContextItem(role="assistant", content=content))

    def add_system(self, content: str):
        self._items.append(ContextItem(role="system", content=content))

    # ── Environment ───────────────────────────────────────────────────────────

    def update_app(self, app: str):
        self.active_app = app

    def update_cwd(self, cwd: str):
        self.cwd = cwd

    def update_screen(self, description: str):
        self.screen_desc = description

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def recent(self, n: int = 10) -> list[ContextItem]:
        items = list(self._items)
        return items[-n:] if len(items) > n else items

    def as_chat_messages(self, n: int = 10) -> list[dict]:
        return [{"role": i.role, "content": i.content} for i in self.recent(n)]

    def summary(self) -> str:
        lines = [
            f"Active app: {self.active_app}",
            f"CWD: {self.cwd}" if self.cwd else "",
            f"Screen: {self.screen_desc}" if self.screen_desc else "",
        ]
        return "\n".join(l for l in lines if l)

    def clear(self):
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
