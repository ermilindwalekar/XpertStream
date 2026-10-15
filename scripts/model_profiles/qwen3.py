"""Qwen3 dense and MoE families: native Qwen <tool_call>{...}</tool_call> format."""

from .qwen import QwenProfile
from .registry import register


@register
class Qwen3Profile(QwenProfile):
    name = "qwen3"
    description = "Qwen3 / Qwen3-Coder: <tool_call>{json}</tool_call>"

    @classmethod
    def matches(cls, model_type, chat_template):
        return model_type == "qwen3" or model_type.startswith("qwen3_")

    def chat_template_kwargs(self):
        return {"enable_thinking": False}

    def generation_defaults(self):
        return {"temperature": 0.7, "top_p": 0.8}