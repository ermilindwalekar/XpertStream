"""Qwen2 / Qwen2.5 (incl. Coder): Hermes-style <tool_call>{...}</tool_call> blocks."""

from .base import ModelProfile, ParsedReply, decode_leading_json, tool_call_from_object
from .registry import register

TOOL_OPEN = "<tool_call>"


@register
class QwenProfile(ModelProfile):
    name = "qwen2"
    description = "Qwen2/Qwen2.5 (incl. Coder): <tool_call>{json}</tool_call>"
    stream_markers = (TOOL_OPEN,)
    tool_format_hint = 'The required format is a <tool_call>{"name": ..., "arguments": {...}}</tool_call> block.'

    @classmethod
    def matches(cls, model_type, chat_template):
        return model_type.startswith("qwen2")

    def parse(self, text, tool_names):
        calls = []
        pos = text.find(TOOL_OPEN)
        first = pos
        while pos >= 0:
            obj, _ = decode_leading_json(text[pos + len(TOOL_OPEN):])
            call = tool_call_from_object(obj, tool_names)
            if call:
                calls.append(call)
            pos = text.find(TOOL_OPEN, pos + len(TOOL_OPEN))
        if calls:
            return ParsedReply(text[:first].strip(), calls)
        # Qwen2.5-Coder frequently ignores the tags and writes ```json blocks or bare JSON instead.
        return self.parse_fallback(text, tool_names)
