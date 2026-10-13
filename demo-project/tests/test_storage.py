import tempfile
import unittest
from pathlib import Path

from taskr.models import Task
from taskr.storage import JsonStorage


class JsonStorageTests(unittest.TestCase):
    def test_missing_file_loads_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(JsonStorage(Path(tmp) / "none.json").load(), [])

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = JsonStorage(Path(tmp) / "tasks.json")
            tasks = [Task(id=1, title="a"), Task(id=2, title="b", priority="low", done=True)]
            storage.save(tasks)
            self.assertEqual(storage.load(), tasks)

    def test_invalid_priority_rejected(self):
        with self.assertRaises(ValueError):
            Task(id=1, title="x", priority="urgent")


if __name__ == "__main__":
    unittest.main()
