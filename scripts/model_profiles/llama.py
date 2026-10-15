"""Llama 3.x: the whole reply is one JSON object {"name": ..., "parameters": {...}}."""

from .base import ModelProfile, ParsedReply, decode_leading_json, tool_call_from_object
from .registry import register

PYTHON_TAG = "<|python_tag|>"


@register
class LlamaProfile(ModelProfile):
    name = "llama3"
    description = 'Llama 3.x: bare {"name": ..., "parameters": {...}} reply'
    tool_format_hint = 'The required format is a single JSON object {"name": ..., "parameters": {...}} and nothing else.'

    @classmethod
    def matches(cls, model_type, chat_template):
        return model_type == "llama" and "<|start_header_id|>" in chat_template

    def prepare_messages(self, messages):
        out = []
        for message in messages:
            calls = message.get("tool_calls", [])
            if len(calls) <= 1:
                out.append(message)
                continue
            for index, call in enumerate(calls):
                out.append({"role": "assistant", "content": message.get("content", "") if index == 0 else "",
                            "tool_calls": [call]})
        return out

    def parse(self, text, tool_names):
        body = text.strip()
        if body.startswith(PYTHON_TAG):
            body = body[len(PYTHON_TAG):]
        obj, _ = decode_leading_json(body)
        call = tool_call_from_object(obj, tool_names)
        if call:
            return ParsedReply("", [call])
        return self.parse_fallback(text, tool_names)

    def tool_call_start(self, text):
        if text.lstrip().startswith(PYTHON_TAG):
            return 0
        return super().tool_call_start(text)

    def holdback(self, text):
        stripped = text.lstrip()
        if stripped and PYTHON_TAG.startswith(stripped):
            return len(text)
        return super().holdback(text)
