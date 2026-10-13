# Taskr

A tiny command-line to-do manager used as a playground for the XpertStream NPU agent.

## Usage

```bash
python main.py add "Buy milk" --priority high
python main.py list
python main.py done 1
python main.py remove 1
```

Tasks are stored in `tasks.json` next to `main.py`.

## Tests

```bash
python -m unittest discover tests
```

## Ideas for the agent

- Add a `--due` date to tasks and show it in `list`.
- Add a `clear-done` command that removes completed tasks.
- Sort `list` output by priority.
- Add a `stats` command (total / done / pending).
