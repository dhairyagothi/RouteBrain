from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TASK_FILES = {
    "easy": DATA_DIR / "task_easy.json",
    "medium": DATA_DIR / "task_medium.json",
    "hard": DATA_DIR / "task_hard.json",
}


class TaskStore:
    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def list_tasks(self) -> list[str]:
        return list(TASK_FILES.keys())

    def get_task(self, task_id: str) -> dict[str, Any]:
        normalized = task_id.lower().strip()
        if normalized not in TASK_FILES:
            raise ValueError(f"Unknown task_id '{task_id}'.")
        if normalized not in self._cache:
            with TASK_FILES[normalized].open("r", encoding="utf-8") as f:
                self._cache[normalized] = json.load(f)
        return self._cache[normalized]


task_store = TaskStore()
