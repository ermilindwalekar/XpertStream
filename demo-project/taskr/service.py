from .models import Task
from .storage import JsonStorage


class TaskNotFound(Exception):
    pass


class TaskService:
    def __init__(self, storage: JsonStorage):
        self.storage = storage
        self.tasks = storage.load()

    def _next_id(self) -> int:
        return max((t.id for t in self.tasks), default=0) + 1

    def _get(self, task_id: int) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise TaskNotFound(f"No task with id {task_id}")

    def add(self, title: str, priority: str = "medium") -> Task:
        task = Task(id=self._next_id(), title=title, priority=priority)
        self.tasks.append(task)
        self.storage.save(self.tasks)
        return task

    def complete(self, task_id: int) -> Task:
        task = self._get(task_id)
        task.done = True
        self.storage.save(self.tasks)
        return task

    def remove(self, task_id: int) -> Task:
        task = self._get(task_id)
        self.tasks.remove(task)
        self.storage.save(self.tasks)
        return task

    def list(self, include_done: bool = True) -> list[Task]:
        if include_done:
            return list(self.tasks)
        return [t for t in self.tasks if not t.done]
