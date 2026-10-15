import tempfile
import unittest
from pathlib import Path

from taskr.service import TaskNotFound, TaskService
from taskr.storage import JsonStorage


# Tests task creation, completion, listing, and missing-task handling.
class TaskServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = TaskService(JsonStorage(Path(self.tmp.name) / "tasks.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_assigns_incrementing_ids(self):
        a = self.service.add("first")
        b = self.service.add("second", "high")
        self.assertEqual((a.id, b.id), (1, 2))
        self.assertEqual(b.priority, "high")

    def test_complete_marks_done(self):
        task = self.service.add("write tests")
        self.service.complete(task.id)
        self.assertTrue(self.service.list()[0].done)
        self.assertEqual(self.service.list(include_done=False), [])

    def test_remove_unknown_raises(self):
        with self.assertRaises(TaskNotFound):
            self.service.remove(42)


if __name__ == "__main__":
    unittest.main()
