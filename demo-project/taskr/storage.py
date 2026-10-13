import json
from pathlib import Path

from .models import Task


class JsonStorage:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> list[Task]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [Task.from_dict(item) for item in json.load(f)]

    def save(self, tasks: list[Task]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in tasks], f, indent=2)
