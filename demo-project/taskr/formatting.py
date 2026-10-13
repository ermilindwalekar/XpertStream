from .models import Task


def format_task(task: Task) -> str:
    mark = "x" if task.done else " "
    return f"[{mark}] {task.id:>3}  {task.priority:<6}  {task.title}"


def format_table(tasks: list[Task]) -> str:
    if not tasks:
        return "No tasks."
    header = "    ID  PRIO    TITLE"
    return "\n".join([header] + [format_task(t) for t in tasks])
