from dataclasses import dataclass, field, asdict
from datetime import datetime

PRIORITIES = ("low", "medium", "high")


@dataclass
class Task:
    id: int
    title: str
    priority: str = "medium"
    done: bool = False
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self):
        if self.priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}, got {self.priority!r}")
        if not self.title.strip():
            raise ValueError("title must not be empty")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(**data)
