"""ai_pc_agent/memory/skill_library.py

Persistent library of automation scripts/skills learned over time.
Supports storing, retrieving, listing, and deleting skills.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from ai_pc_agent.utils import config
from ai_pc_agent.utils.logger import get_logger

logger = get_logger("agent.skills")


@dataclass
class Skill:
    name:        str
    description: str
    code:        str
    trigger:     str            # wake phrase or intent that triggers it
    use_count:   int   = 0
    ts_created:  float = field(default_factory=time.time)
    ts_updated:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Skill":
        return Skill(**d)


class SkillLibrary:
    """JSON-backed store of reusable automation skills."""

    def __init__(self, file_path: str | None = None):
        fp = file_path or config.get("SKILL_LIBRARY_FILE", "skill_library.json")
        self.path = Path(fp)
        self._skills: dict[str, Skill] = {}
        self._load()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, name: str, description: str, code: str, trigger: str = "") -> Skill:
        skill = Skill(name=name, description=description, code=code, trigger=trigger)
        self._skills[name.lower()] = skill
        self._save()
        logger.info("Skill added: '%s'", name)
        return skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name.lower())

    def find_by_trigger(self, text: str) -> Skill | None:
        text_lower = text.lower()
        for skill in self._skills.values():
            if skill.trigger and skill.trigger.lower() in text_lower:
                return skill
        return None

    def delete(self, name: str) -> bool:
        key = name.lower()
        if key in self._skills:
            del self._skills[key]
            self._save()
            logger.info("Skill deleted: '%s'", name)
            return True
        return False

    def update_code(self, name: str, code: str) -> bool:
        skill = self._skills.get(name.lower())
        if skill:
            skill.code       = code
            skill.ts_updated = time.time()
            self._save()
            return True
        return False

    def increment_use(self, name: str):
        skill = self._skills.get(name.lower())
        if skill:
            skill.use_count += 1
            self._save()

    # ── Listing ───────────────────────────────────────────────────────────────

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def most_used(self, n: int = 5) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.use_count, reverse=True)[:n]

    def summary_lines(self) -> list[str]:
        return [
            f"[{k}] {v.description} (used {v.use_count}×)"
            for k, v in self._skills.items()
        ]

    def __len__(self) -> int:
        return len(self._skills)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = {k: v.to_dict() for k, v in self._skills.items()}
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("SkillLibrary save failed: %s", exc)

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._skills = {k: Skill.from_dict(v) for k, v in data.items()}
            logger.info("Loaded %d skill(s) from %s", len(self._skills), self.path)
        except Exception as exc:
            logger.warning("SkillLibrary load failed: %s", exc)
