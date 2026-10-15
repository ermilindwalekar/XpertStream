"""Base class and shared helpers for model profiles.

A profile captures everything the agent server must know about a model family that is not
covered by its Hugging Face chat template: how the model writes tool calls, how to spot a tool
call while streaming, and any extra prompting or tokenizer checks it needs.
"""

import json
import uuid
from dataclasses import dataclass, field

BASE_TOOL_RULES = """# How to act
- Do the task yourself by calling tools. Do not describe a plan or explain what you are going to do.
- Each reply is either exactly one tool call in the required format, or a short final answer once the task is done.
- Never write tool calls inside ``` code blocks.
- Never invent file contents: read a file before changing it, and wait for each tool result before the next call."""

# Code a detokenizer must round-trip unchanged; "clean up tokenization spaces" turns it into "from.models".
DETOKENIZER_PROBE = "from .models import Task\nx = a .b , c"


# Where the fallback formats (bare JSON on its own line, ```json blocks) start; any model may use them.
FALLBACK_MARKERS = ("\n{", "```json")


@dataclass
class ParsedReply:
    content: str
    tool_calls: list = field(default_factory=list)


def make_tool_call(name: str, arguments) -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_call_from_object(obj, tool_names: set):
    """Builds an OpenAI tool call from {"name", "arguments"|"parameters"}; None if it isn't a known tool."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or (tool_names and name not in tool_names):
        return None
    return make_tool_call(name, obj.get("arguments", obj.get("parameters", {})))


def decode_leading_json(text: str):
    """Decodes the JSON object at the start of `text` (ignoring trailing text); returns (obj, end) or (None, 0)."""
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return None, 0
    try:
        obj, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None, 0
    return obj, len(text) - len(stripped) + end


def held_back_length(text: str, marker: str) -> int:
    """Length of the longest suffix of `text` that could be the start of `marker`."""
    for n in range(min(len(marker) - 1, len(text)), 0, -1):
        if marker.startswith(text[-n:]):
            return n
    return 0


class ModelProfile:
    """Default behaviour; subclasses override what differs for their model family."""

    name = "generic"
    description = "Fallback for unknown models: fenced or bare JSON tool calls"
    # Text that starts a tool call in this family's native format, in addition to FALLBACK_MARKERS.
    stream_markers: tuple = ()
    tool_format_hint = 'The required format is a single JSON object {"name": ..., "arguments": {...}} and nothing else.'

    @classmethod
    def matches(cls, model_type: str, chat_template: str) -> bool:
        """True if this profile handles a model with this config.json model_type and chat template."""
        return False

    def tool_rules(self) -> str:
        return f"{BASE_TOOL_RULES}\n- {self.tool_format_hint}"

    def chat_template_kwargs(self) -> dict:
        return {}

    def generation_defaults(self) -> dict:
        return {"temperature": 0.0, "top_p": 0.95}

    def prepare_messages(self, messages: list) -> list:
        return messages

    def parse(self, text: str, tool_names: set) -> ParsedReply:
        return self.parse_fallback(text, tool_names)

    def parse_fallback(self, text: str, tool_names: set) -> ParsedReply:
        """Formats small models use when they ignore their native one: ```json blocks or a bare JSON reply."""
        if not tool_names:
            return ParsedReply(text.strip())
        start = text.find("```")
        while start >= 0:
            body_start = text.find("\n", start)
            end = text.find("```", body_start + 1) if body_start >= 0 else -1
            if end < 0:
                break
            obj, _ = decode_leading_json(text[body_start + 1:end])
            call = tool_call_from_object(obj, tool_names)
            if call:
                # Several fenced calls are usually a written-out plan; later ones were guessed before seeing results.
                return ParsedReply(text[:start].strip(), [call])
            start = text.find("```", end + 3)
        obj, _ = decode_leading_json(text)
        call = tool_call_from_object(obj, tool_names)
        if call:
            return ParsedReply("", [call])
        return self.parse_trailing(text, tool_names)

    def parse_trailing(self, text: str, tool_names: set) -> ParsedReply:
        """A tool call written as a JSON object on its own line at the very end, after some prose."""
        start = text.find("\n{")
        while start >= 0:
            obj, end = decode_leading_json(text[start + 1:])
            call = tool_call_from_object(obj, tool_names)
            # Only the tail of the reply counts; JSON in the middle of an explanation is not a call.
            if call and not text[start + 1 + end:].strip():
                return ParsedReply(text[:start].strip(), [call])
            start = text.find("\n{", start + 2)
        return ParsedReply(text.strip())

    def tool_call_start(self, text: str):
        """Index where a tool call begins in streamed text, or None if none has started."""
        if text.lstrip().startswith("{"):
            return 0
        found = [i for i in (text.find(m) for m in (*self.stream_markers, *FALLBACK_MARKERS)) if i >= 0]
        return min(found) if found else None

    def holdback(self, text: str) -> int:
        """Characters at the end of `text` not to stream yet because they may begin a tool call."""
        # Until the first real character we can't tell whether the whole reply is a bare JSON call.
        if not text.strip():
            return len(text)
        return max((held_back_length(text, m) for m in (*self.stream_markers, *FALLBACK_MARKERS)), default=0)

    def check_detokenizer(self, decode, encode) -> list:
        """Returns warnings if the OpenVINO detokenizer alters code; `encode`/`decode` map text <-> token ids."""
        decoded = decode(encode(DETOKENIZER_PROBE))
        if decoded != DETOKENIZER_PROBE:
            return [f"Detokenizer alters code: {DETOKENIZER_PROBE!r} -> {decoded!r}. "
                    "Fix with: python scripts\\export_npu_model.py --fix-detokenizer <model_dir>"]
        return []
