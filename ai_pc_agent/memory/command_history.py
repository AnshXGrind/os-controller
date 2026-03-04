"""ai_pc_agent/memory/command_history.py

Stores and replays command history.  Optionally persisted to JSON.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.history")


@dataclass
class HistoryEntry:
    raw_text: str
    intent:   str
    value:    str
    success:  bool
    ts:       float = field(default_factory=time.time)


class CommandHistory:
    """Ring-buffer style command history with optional JSON persistence."""

    def __init__(self, max_size: int = 100, persist_file: str | None = None):
        self.max_size     = max_size
        self.persist_file = Path(persist_file) if persist_file else None
        self._entries:    list[HistoryEntry] = []
        if self.persist_file and self.persist_file.exists():
            self._load()

    # ── Add ───────────────────────────────────────────────────────────────────

    def add(self, raw_text: str, intent: str, value: str, success: bool):
        entry = HistoryEntry(raw_text=raw_text, intent=intent, value=value, success=success)
        self._entries.append(entry)
        if len(self._entries) > self.max_size:
            self._entries.pop(0)
        if self.persist_file:
            self._save()

    # ── Query ─────────────────────────────────────────────────────────────────

    def last(self) -> HistoryEntry | None:
        return self._entries[-1] if self._entries else None

    def last_successful(self) -> HistoryEntry | None:
        for e in reversed(self._entries):
            if e.success:
                return e
        return None

    def recent(self, n: int = 5) -> list[HistoryEntry]:
        return list(reversed(self._entries[-n:]))

    def find_by_intent(self, intent: str) -> list[HistoryEntry]:
        return [e for e in self._entries if e.intent == intent]

    def summary_lines(self, n: int = 10) -> list[str]:
        lines = []
        for e in self.recent(n):
            status = "✓" if e.success else "✗"
            lines.append(f"{status} [{e.intent}] {e.raw_text}")
        return lines

    def clear(self):
        self._entries.clear()
        if self.persist_file:
            self._save()

    def __len__(self) -> int:
        return len(self._entries)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = [asdict(e) for e in self._entries]
            self.persist_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("History save failed: %s", exc)

    def _load(self):
        try:
            data = json.loads(self.persist_file.read_text(encoding="utf-8"))
            self._entries = [HistoryEntry(**d) for d in data]
            logger.info("Loaded %d history entries from %s", len(self._entries), self.persist_file)
        except Exception as exc:
            logger.warning("History load failed: %s", exc)
