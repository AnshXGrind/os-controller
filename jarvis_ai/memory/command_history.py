"""
memory/command_history.py

Persistent in-memory command history with optional JSON file backing.

Supports:
    • Storing every executed (intent, value, raw_text) tuple
    • Retrieving the last command for "repeat last command"
    • Listing recent N commands
    • Clearing history
    • Persisting to / loading from a JSON file (optional)
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_FILE = "jarvis_history.json"
MAX_IN_MEMORY        = 200   # cap to avoid unbounded growth


@dataclass
class HistoryEntry:
    """One command execution record."""
    timestamp:  float     # Unix timestamp
    raw_text:   str       # original spoken text
    intent:     str       # parsed intent string
    value:      str       # parsed value / "" if none
    success:    bool      # whether the action succeeded

    def as_readable(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        v  = f" → {self.value}" if self.value else ""
        ok = "✓" if self.success else "✗"
        return f"[{ts}] {ok} {self.intent}{v}  ('{self.raw_text}')"


class CommandHistory:
    """
    Records and retrieves command execution history.

    Args:
        max_entries:  Maximum number of entries kept in memory.
        persist_file: Path to a JSON file for optional persistence.
                      Pass None to keep history in memory only.
    """

    def __init__(
        self,
        max_entries:  int           = MAX_IN_MEMORY,
        persist_file: Optional[str] = None,
    ):
        self.max_entries  = max_entries
        self._file        = Path(persist_file) if persist_file else None
        self._history:    list[HistoryEntry] = []
        self._lock        = threading.Lock()

        if self._file:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        raw_text: str,
        intent:   str,
        value:    str  = "",
        success:  bool = True,
    ) -> HistoryEntry:
        """
        Record a command execution.

        Args:
            raw_text: The full recognised speech string.
            intent:   Parsed intent identifier.
            value:    Optional target value (app name, query, etc.).
            success:  Whether the action completed without error.

        Returns:
            The created HistoryEntry.
        """
        entry = HistoryEntry(
            timestamp = time.time(),
            raw_text  = raw_text,
            intent    = intent,
            value     = value,
            success   = success,
        )
        self._history.append(entry)

        # Trim if over limit
        if len(self._history) > self.max_entries:
            self._history = self._history[-self.max_entries:]

        if self._file:
            self._save()

        return entry

    def last(self) -> Optional[HistoryEntry]:
        """Return the most recent entry, or None if history is empty."""
        return self._history[-1] if self._history else None

    def recent(self, n: int = 10) -> list[HistoryEntry]:
        """Return up to the last *n* entries, newest last."""
        return self._history[-n:]

    def last_successful(self) -> Optional[HistoryEntry]:
        """Return the most recent entry that was marked as successful."""
        for entry in reversed(self._history):
            if entry.success:
                return entry
        return None

    def clear(self):
        """Wipe all history."""
        self._history.clear()
        if self._file and self._file.exists():
            self._file.unlink()

    def summary_lines(self, n: int = 10) -> list[str]:
        """Return human-readable lines for the last *n* commands."""
        return [e.as_readable() for e in self.recent(n)]

    def __len__(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        with self._lock:
            try:
                data = [asdict(e) for e in self._history]
                self._file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.warning(f"CommandHistory: could not save to '{self._file}': {exc}")

    def _load(self):
        with self._lock:
            if not self._file.exists():
                return
            try:
                raw = json.loads(self._file.read_text(encoding="utf-8"))
                self._history = [HistoryEntry(**r) for r in raw]
                logger.info(f"CommandHistory: loaded {len(self._history)} entries from '{self._file}'.")
            except Exception as exc:
                logger.warning(f"CommandHistory: could not load '{self._file}': {exc}")
