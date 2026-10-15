import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from configure_opencode import fetch_model, make_config, register_global_config


class MockResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value).encode()


class ConfigTests(unittest.TestCase):
    def test_uses_active_local_model_and_preserves_existing_settings(self):
        existing = {"theme": "custom", "provider": {"other": {"models": {"a": {}}}}}
        health = {"status": "ok", "device": "NPU", "max_prompt_len": 4096, "max_response_len": 512}
        result = make_config(existing, health, "llama3.1-npu", "http://127.0.0.1:8765")
        self.assertEqual(result["provider"]["other"], existing["provider"]["other"])
        self.assertEqual(result["provider"]["xpertstream"]["options"]["baseURL"], "http://127.0.0.1:8765/v1")
        self.assertEqual(result["provider"]["xpertstream"]["models"]["llama3.1-npu"]["limit"],
                         {"context": int(4096 * 0.85) + 512, "output": 512})
        self.assertEqual(result["small_model"], "xpertstream/llama3.1-npu")
        self.assertEqual(result["default_agent"], "xpert")
        self.assertEqual(result["agent"]["xpert"]["model"], "xpertstream/llama3.1-npu")
        self.assertEqual(result["agent"]["xpert"]["permission"]["bash"], "ask")
        self.assertEqual(result["agent"]["xpert"]["permission"]["task"], "deny")
        self.assertEqual(result["share"], "disabled")
        self.assertEqual(result["enabled_providers"], ["xpertstream"])
        self.assertEqual(existing["theme"], "custom")

    def test_detects_model_from_server_not_hardcoded_folder(self):
        responses = [MockResponse({"status": "ok", "device": "GPU", "max_prompt_len": 8192,
                                   "max_response_len": 1024}),
                     MockResponse({"data": [{"id": "qwen2.5-coder-7b-npu"}]})]
        with patch("configure_opencode.urlopen", side_effect=responses) as fetch:
            health, model = fetch_model("http://127.0.0.1:8765")
        self.assertEqual(model, "qwen2.5-coder-7b-npu")
        self.assertEqual(health["device"], "GPU")
        self.assertEqual(fetch.call_args_list[1].args[0], "http://127.0.0.1:8765/v1/models")

    def test_unhealthy_server_is_not_registered(self):
        with patch("configure_opencode.urlopen", return_value=MockResponse({"status": "error", "error": "GPU allocation"})):
            with self.assertRaisesRegex(ValueError, "GPU allocation"):
                fetch_model("http://127.0.0.1:8765")

    def test_registers_and_updates_only_our_local_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opencode.json"
            health = {"status": "ok", "device": "NPU", "max_prompt_len": 16384, "max_response_len": 512}
            self.assertTrue(register_global_config(health, "qwen2.5-npu", "http://127.0.0.1:8765", path))
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["model"], "xpertstream/qwen2.5-npu")
            self.assertEqual(config["provider"]["xpertstream"]["options"]["baseURL"], "http://127.0.0.1:8765/v1")
            health["device"] = "GPU.0"
            self.assertTrue(register_global_config(health, "llama3.1-npu", "http://127.0.0.1:8765", path))
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["model"], "xpertstream/llama3.1-npu")
            self.assertEqual(config["provider"]["xpertstream"]["models"]["llama3.1-npu"]["name"], "llama3.1-npu (GPU.0)")

    def test_does_not_overwrite_an_existing_unrelated_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opencode.json"
            path.write_text('{"model": "another-provider/model"}', encoding="utf-8")
            original = path.read_bytes()
            self.assertFalse(register_global_config({"device": "NPU", "max_prompt_len": 8192,
                                                     "max_response_len": 512}, "qwen2.5-npu",
                                                    "http://127.0.0.1:8765", path))
            self.assertEqual(path.read_bytes(), original)

    def test_does_not_overwrite_existing_jsonc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opencode.json"
            path.with_suffix(".jsonc").write_text('{ // user settings\n}', encoding="utf-8")
            self.assertFalse(register_global_config({"device": "NPU", "max_prompt_len": 8192,
                                                     "max_response_len": 512}, "qwen2.5-npu",
                                                    "http://127.0.0.1:8765", path))
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()