import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from export_npu_model import MODEL_PRESETS, export_model_for_npu
from model_profiles import select_profile


class Qwen3MoeExportTests(unittest.TestCase):
    def test_native_crash_is_not_reported_as_authentication_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            for returncode in (3221225477, -1073741819):
                with self.subTest(returncode=returncode), patch("export_npu_model.subprocess.run") as run:
                    run.return_value.returncode = returncode
                    output = io.StringIO()
                    with redirect_stdout(output), self.assertRaises(SystemExit):
                        export_model_for_npu("Qwen/Qwen3-30B-A3B", directory)
                    self.assertIn("0xC0000005", output.getvalue())
                    self.assertNotIn("401 Unauthorized", output.getvalue())

    def test_qwen3_30b_a3b_preset_and_export_command(self):
        preset = MODEL_PRESETS["qwen3-30b-a3b"]
        self.assertEqual(preset["model_id"], "Qwen/Qwen3-30B-A3B")
        self.assertEqual(preset["default_dir"], "./models/qwen3-30b-a3b-npu")

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            with patch("export_npu_model.subprocess.run") as run, patch("export_npu_model.rebuild_detokenizer"):
                run.return_value.returncode = 0
                export_model_for_npu(preset["model_id"], directory, max_seq_len=16384)

            command = run.call_args.args[0]
            self.assertEqual(command[command.index("openvino") + 2], preset["model_id"])
            self.assertEqual(command[command.index("--weight-format") + 1], "int4")
            self.assertIn(str(model_dir.resolve()), command)
            metadata = json.loads((model_dir / "npu_static_config.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["max_seq_len"], 16384)

            (model_dir / "config.json").write_text('{"model_type": "qwen3_moe"}', encoding="utf-8")
            self.assertEqual(select_profile(model_dir, "").name, "qwen3")


if __name__ == "__main__":
    unittest.main()