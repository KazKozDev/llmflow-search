"""Persistent research memory store (strategies, experiences, skills)."""

import json
import os
import re
from datetime import datetime
from pathlib import Path


def _slug_key(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:80] or "item"


class ResearchMemoryStore:
    """Persistent JSON-backed strategy/skill memory for research tasks."""

    def __init__(self, path: str | None = None):
        default_path = os.getenv(
            "LLMFLOW_SEARCH_RESEARCH_MEMORY", "~/.llmflow-search/research_memory.json"
        )
        self.path = Path(path or default_path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if isinstance(data, dict):
                    return {
                        "strategies": data.get("strategies", {}),
                        "skills": data.get("skills", {}),
                        "experiences": data.get("experiences", []),
                    }
            except (OSError, json.JSONDecodeError):
                pass
        return {"strategies": {}, "skills": {}, "experiences": []}

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, default=str)
        )

    def get_strategies(self, limit: int = 20) -> list[dict]:
        strategies = list(self.data.get("strategies", {}).values())
        strategies.sort(
            key=lambda item: (
                item.get("success_rate", 0.0),
                item.get("wins", 0),
                item.get("plays", 0),
            ),
            reverse=True,
        )
        return strategies[:limit]

    def best_strategy(
        self, min_plays: int = 1, min_success_rate: float = 0.6
    ) -> dict | None:
        for strategy in self.get_strategies():
            if (
                strategy.get("plays", 0) >= min_plays
                and strategy.get("success_rate", 0.0) >= min_success_rate
            ):
                return strategy
        return None

    def record_strategy(
        self, desc: str, success: bool, won: bool = False, meta: dict | None = None
    ) -> dict:
        if not desc:
            return {}
        key = _slug_key(desc)
        strategies = self.data.setdefault("strategies", {})
        strategy = strategies.get(key) or {
            "key": key,
            "desc": desc,
            "plays": 0,
            "wins": 0,
            "success_rate": 0.5,
        }
        strategy["plays"] = strategy.get("plays", 0) + 1
        if won:
            strategy["wins"] = strategy.get("wins", 0) + 1
        old = float(strategy.get("success_rate", 0.5))
        strategy["success_rate"] = round(old * 0.7 + (1.0 if success else 0.0) * 0.3, 3)
        strategy["last_used"] = datetime.now().isoformat(timespec="seconds")
        if meta:
            strategy["last_meta"] = meta
        strategies[key] = strategy
        self._save()
        return strategy

    def add_experience(self, exp: dict) -> None:
        experiences = self.data.setdefault("experiences", [])
        exp["timestamp"] = datetime.now().isoformat(timespec="seconds")
        experiences.append(exp)
        self.data["experiences"] = experiences[-500:]
        self._save()

    def save_skill(self, skill: dict) -> None:
        name = skill.get("name") or _slug_key(skill.get("trigger", "research-skill"))
        skill["name"] = name
        skill["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.data.setdefault("skills", {})[name] = skill
        self._save()

    def get_skills(self, limit: int = 10) -> list[dict]:
        skills = list(self.data.get("skills", {}).values())
        skills.sort(
            key=lambda item: (item.get("success_rate", 0.0), item.get("use_count", 0)),
            reverse=True,
        )
        return skills[:limit]


_research_store: ResearchMemoryStore | None = None


def _get_research_store() -> ResearchMemoryStore:
    global _research_store
    if _research_store is None:
        _research_store = ResearchMemoryStore()
    return _research_store
