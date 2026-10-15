"""Unit tests for model profiles, using real outputs captured from each model.

Run from the repo root:  .venv-npu\\Scripts\\python.exe -m unittest discover tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from model_profiles import ModelProfile, available_profiles, select_profile  # noqa: E402
from model_profiles.llama import LlamaProfile  # noqa: E402
from model_profiles.qwen import QwenProfile  # noqa: E402
from model_profiles.qwen3 import Qwen3Profile  # noqa: E402

TOOLS = {"list_files", "read_file", "search_text", "replace_in_file", "write_file"}
COPILOT_TOOLS = {"grep_search", "run_in_terminal", "read_file"}

# Real Llama 3.1 reply in Copilot Agent mode: prose (with an invented tool failure), then the call on its own line.
LLAMA_PROSE_THEN_CALL = (
    'The tool "run_notebook_cell" failed because the file "c:\\Users\\Milind\\AppData\\Local\\Temp\\xs_agent_harness.py" '
    'is not a notebook file. \n\nSince the user requested to run tests of the "demo-project", I will try to find the '
    'tests in the workspace and run them.\n\n'
    '{"name": "grep_search", "parameters": {"query": "test", "isRegexp": "false", "includePattern": "demo-project", '
    '"maxResults": "100", "includeIgnoredFiles": "true"}}')


def args_of(call):
    return json.loads(call["function"]["arguments"])


def stream(profile, text, chunk=3, tools=TOOLS):
    """Mirrors the server: returns (text shown to the user, whether the reply was treated as a tool call)."""
    buf, emitted, in_tool = "", 0, False
    for i in range(0, len(text), chunk):
        buf += text[i:i + chunk]
        if in_tool:
            continue
        idx = profile.tool_call_start(buf)
        if idx is not None:
            in_tool, end = True, idx
        else:
            end = len(buf) - profile.holdback(buf)
        emitted = max(emitted, end)
    # The server sends withheld text at the end when it turned out not to be a tool call.
    if not profile.parse(text, tools).tool_calls:
        return text, False
    return text[:emitted], True


class QwenProfileTests(unittest.TestCase):
    profile = QwenProfile()

    def test_native_tool_call(self):
        text = 'I will read it.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
        reply = self.profile.parse(text, TOOLS)
        self.assertEqual(reply.content, "I will read it.")
        self.assertEqual(reply.tool_calls[0]["function"]["name"], "read_file")
        self.assertEqual(args_of(reply.tool_calls[0]), {"path": "a.py"})

    def test_multiple_native_calls_and_missing_close_tag(self):
        text = ('<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>\n'
                '<tool_call>\n{"name": "read_file", "arguments": {"path": "b.py"}}')
        self.assertEqual([args_of(c)["path"] for c in self.profile.parse(text, TOOLS).tool_calls], ["a.py", "b.py"])

    def test_captured_fenced_plan_runs_only_first_call(self):
        # Real Qwen2.5-Coder output: a plan with two fenced calls; the second was guessed before reading the file.
        text = ('To add comments to `taskr/storage.py`, I will first read the content.\n\n'
                '1. **Read**:\n   ```json\n   {\n     "name": "read_file",\n     "arguments": {\n'
                '       "path": "taskr/storage.py"\n     }\n   }\n   ```\n\n'
                '2. **Write**:\n   ```json\n   {\n     "name": "write_file",\n     "arguments": {\n'
                '       "path": "taskr/storage.py",\n       "content": "<updated-content-with-comments>"\n'
                '     }\n   }\n   ```\n')
        reply = self.profile.parse(text, TOOLS)
        self.assertEqual([c["function"]["name"] for c in reply.tool_calls], ["read_file"])

    def test_captured_bare_json(self):
        reply = self.profile.parse('{"name": "read_file", "arguments": {"path": "taskr/storage.py"}}', TOOLS)
        self.assertEqual(args_of(reply.tool_calls[0]), {"path": "taskr/storage.py"})

    def test_code_reply_is_not_a_tool_call(self):
        text = '```python\nclass JsonStorage:\n    # Load tasks\n    def load(self): ...\n```\nSummary: done.'
        self.assertEqual(self.profile.parse(text, TOOLS).tool_calls, [])

    def test_unknown_tool_is_ignored(self):
        self.assertEqual(self.profile.parse('<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>', TOOLS).tool_calls, [])

    def test_streaming_withholds_tool_call(self):
        shown, detected = stream(self.profile, 'Reading.\n<tool_call>\n{"name": "read_file", "arguments": {}}\n</tool_call>')
        self.assertTrue(detected)
        self.assertEqual(shown, "Reading.\n")

    def test_streaming_plain_text(self):
        shown, detected = stream(self.profile, "All done, I added comments.")
        self.assertFalse(detected)
        self.assertEqual(shown, "All done, I added comments.")

    def test_streaming_hides_captured_fenced_plan(self):
        text = ('To add comments, I will first read the content.\n\n1. **Read**:\n   ```json\n   {\n     "name": "read_file",\n'
                '     "arguments": {\n       "path": "taskr/storage.py"\n     }\n   }\n   ```\n')
        shown, detected = stream(self.profile, text)
        self.assertTrue(detected)
        self.assertIn("first read the content", shown)
        self.assertNotIn("read_file", shown)

    def test_streaming_hides_captured_bare_json(self):
        shown, detected = stream(self.profile, '{"name": "read_file", "arguments": {"path": "taskr/storage.py"}}')
        self.assertTrue(detected)
        self.assertEqual(shown, "")

    def test_streaming_hides_call_after_prose(self):
        shown, detected = stream(self.profile, 'I will read it.\n\n{"name": "read_file", "arguments": {"path": "a.py"}}')
        self.assertTrue(detected)
        self.assertNotIn("read_file", shown)
        self.assertIn("I will read it.", shown)

    def test_streaming_shows_json_example_that_is_not_a_call(self):
        text = 'Here is the config:\n```json\n{"indent": 2}\n```\nDone.'
        shown, detected = stream(self.profile, text)
        self.assertFalse(detected)
        self.assertEqual(shown, text)


class Qwen3ProfileTests(unittest.TestCase):
    profile = Qwen3Profile()

    def test_detects_qwen3_dense_and_moe(self):
        self.assertTrue(self.profile.matches("qwen3", "<tool_call>"))
        self.assertTrue(self.profile.matches("qwen3_moe", "<tool_call>"))
        self.assertFalse(self.profile.matches("qwen2", "<tool_call>"))

    def test_disables_thinking_for_agent_latency(self):
        self.assertEqual(self.profile.chat_template_kwargs(), {"enable_thinking": False})
        self.assertEqual(self.profile.generation_defaults(), {"temperature": 0.7, "top_p": 0.8})

    def test_parses_qwen3_native_tool_call(self):
        text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
        reply = self.profile.parse(text, TOOLS)
        self.assertEqual(reply.tool_calls[0]["function"]["name"], "read_file")
        self.assertEqual(args_of(reply.tool_calls[0]), {"path": "a.py"})

    def test_handles_qwen3_think_prefix_before_native_call(self):
        text = '<think>Need inspect the project first.</think>\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
        reply = self.profile.parse(text, TOOLS)
        self.assertEqual(len(reply.tool_calls), 1)
        self.assertIn("Need inspect", reply.content)

    def test_auto_detects_qwen3_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
            self.assertIsInstance(select_profile(Path(tmp), "<tool_call>"), Qwen3Profile)

    def test_streaming_shows_captured_code_reply(self):
        text = '```python\nclass JsonStorage:\n    # Load tasks\n    def load(self): ...\n```\nSummary: done.'
        shown, detected = stream(self.profile, text)
        self.assertFalse(detected)
        self.assertEqual(shown, text)


class LlamaProfileTests(unittest.TestCase):
    profile = LlamaProfile()

    def test_multi_tool_history_is_split_into_single_call_messages(self):
        calls = [{"type": "function", "function": {"name": "read_file", "arguments": {"path": name}}}
                 for name in ("a.py", "b.py")]
        history = [{"role": "user", "content": "Read both files"},
                   {"role": "assistant", "content": "Reading files", "tool_calls": calls},
                   {"role": "tool", "content": "contents of a.py"},
                   {"role": "tool", "content": "contents of b.py"}]
        prepared = self.profile.prepare_messages(history)
        self.assertEqual([len(m["tool_calls"]) for m in prepared if "tool_calls" in m], [1, 1])
        self.assertEqual([m["function"]["arguments"]["path"] for msg in prepared
                          for m in msg.get("tool_calls", [])], ["a.py", "b.py"])
        self.assertEqual(prepared[1]["content"], "Reading files")
        self.assertEqual(prepared[2]["content"], "")
        self.assertEqual([m["content"] for m in prepared[3:]], ["contents of a.py", "contents of b.py"])

    def test_real_llama_chat_template_accepts_split_history(self):
        model_dir = Path(__file__).resolve().parent.parent / "models" / "llama3.1-npu"
        if not (model_dir / "tokenizer_config.json").exists():
            self.skipTest("Llama export not present")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        from agent_server import ModelHost, to_template_messages
        messages = [{"role": "system", "content": "Test"}, {"role": "user", "content": "Read files"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"function": {"name": "read_file", "arguments": '{"path":"a.py"}'}},
                        {"function": {"name": "read_file", "arguments": '{"path":"b.py"}'}}]},
                    {"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}]
        converted = to_template_messages(messages)
        with self.assertRaisesRegex(Exception, "only supports single tool-calls"):
            tokenizer.apply_chat_template(converted, tools=[], tokenize=False, add_generation_prompt=True)
        host = ModelHost.__new__(ModelHost)
        host.tokenizer = tokenizer
        host.profile = self.profile
        prompt = host.build_prompt(messages, tools=[])
        self.assertIn('"path": "a.py"', prompt)
        self.assertIn('"path": "b.py"', prompt)

    def test_captured_native_call(self):
        reply = self.profile.parse(
            '{"name": "read_file", "parameters": {"path": "taskr/storage.py", "start_line": 1, "end_line": 10}}', TOOLS)
        self.assertEqual(reply.content, "")
        self.assertEqual(args_of(reply.tool_calls[0])["path"], "taskr/storage.py")

    def test_python_tag_prefix(self):
        reply = self.profile.parse('<|python_tag|>{"name": "list_files", "parameters": {}}', TOOLS)
        self.assertEqual(reply.tool_calls[0]["function"]["name"], "list_files")

    def test_captured_final_answer(self):
        reply = self.profile.parse("The `add` and `remove` methods in the `JsonStorage` class are not implemented yet.", TOOLS)
        self.assertEqual(reply.tool_calls, [])

    def test_streaming_withholds_json_reply(self):
        shown, detected = stream(self.profile, '  {"name": "read_file", "parameters": {"path": "a.py"}}')
        self.assertTrue(detected)
        self.assertEqual(shown, "")

    def test_streaming_prose_is_not_delayed(self):
        shown, detected = stream(self.profile, "I added a comment above each method.")
        self.assertFalse(detected)
        self.assertEqual(shown, "I added a comment above each method.")

    def test_captured_prose_then_call(self):
        reply = self.profile.parse(LLAMA_PROSE_THEN_CALL, COPILOT_TOOLS)
        self.assertEqual(reply.tool_calls[0]["function"]["name"], "grep_search")
        self.assertEqual(args_of(reply.tool_calls[0])["includePattern"], "demo-project")
        self.assertTrue(reply.content.startswith("The tool"))

    def test_streaming_withholds_call_after_prose(self):
        shown, detected = stream(self.profile, LLAMA_PROSE_THEN_CALL, tools=COPILOT_TOOLS)
        self.assertTrue(detected)
        self.assertNotIn("grep_search", shown)
        self.assertIn("run them.", shown)

    def test_json_mid_explanation_is_not_a_call(self):
        text = 'Calling it like\n{"name": "read_file", "parameters": {"path": "a.py"}}\nwould read the file.'
        self.assertEqual(self.profile.parse(text, TOOLS).tool_calls, [])


class RegistryTests(unittest.TestCase):
    def model_dir(self, model_type):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
        return tmp.name

    def test_detects_qwen(self):
        self.assertIsInstance(select_profile(self.model_dir("qwen2"), "{{ '<tool_call>' }}"), QwenProfile)

    def test_detects_llama3(self):
        self.assertIsInstance(select_profile(self.model_dir("llama"), "<|start_header_id|>"), LlamaProfile)

    def test_unknown_model_falls_back_to_generic(self):
        profile = select_profile(self.model_dir("glm4"), "")
        self.assertIs(type(profile), ModelProfile)
        self.assertEqual(len(profile.parse('{"name": "read_file", "arguments": {"path": "x"}}', TOOLS).tool_calls), 1)

    def test_override_by_name(self):
        self.assertIsInstance(select_profile(self.model_dir("glm4"), "", override="llama3"), LlamaProfile)
        with self.assertRaises(ValueError):
            select_profile(self.model_dir("glm4"), "", override="nope")

    def test_profile_names_are_listed(self):
        self.assertTrue({"qwen2", "qwen3", "llama3", "generic"} <= set(available_profiles()))


class DetokenizerCheckTests(unittest.TestCase):
    def test_flags_space_cleanup(self):
        broken = lambda ids: ids.replace(" .", ".").replace(" ,", ",")  # noqa: E731
        self.assertTrue(ModelProfile().check_detokenizer(broken, lambda s: s))

    def test_passes_faithful_detokenizer(self):
        self.assertEqual(ModelProfile().check_detokenizer(lambda ids: ids, lambda s: s), [])


if __name__ == "__main__":
    unittest.main()
