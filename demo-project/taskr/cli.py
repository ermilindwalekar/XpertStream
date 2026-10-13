import argparse
from pathlib import Path

from .formatting import format_table, format_task
from .models import PRIORITIES
from .service import TaskNotFound, TaskService
from .storage import JsonStorage

DEFAULT_DB = Path(__file__).resolve().parent.parent / "tasks.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="taskr", description="Tiny to-do manager")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to the tasks JSON file")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Add a task")
    add.add_argument("title")
    add.add_argument("--priority", choices=PRIORITIES, default="medium")

    lst = sub.add_parser("list", help="List tasks")
    lst.add_argument("--pending", action="store_true", help="Hide completed tasks")

    done = sub.add_parser("done", help="Mark a task as done")
    done.add_argument("id", type=int)

    remove = sub.add_parser("remove", help="Remove a task")
    remove.add_argument("id", type=int)
    return parser


def run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    service = TaskService(JsonStorage(args.db))
    try:
        if args.command == "add":
            print("Added:", format_task(service.add(args.title, args.priority)))
        elif args.command == "list":
            print(format_table(service.list(include_done=not args.pending)))
        elif args.command == "done":
            print("Completed:", format_task(service.complete(args.id)))
        elif args.command == "remove":
            print("Removed:", format_task(service.remove(args.id)))
    except TaskNotFound as e:
        print(f"Error: {e}")
        return 1
    return 0
