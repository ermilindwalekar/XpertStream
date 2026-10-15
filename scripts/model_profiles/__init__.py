"""Model profiles: per-family tool-call formats and quirks for the agent server.

To support a new model family, add a module in this package that defines a ModelProfile
subclass decorated with @register (see qwen.py / llama.py). It is picked up automatically.
"""

import importlib
import pkgutil

from .base import ModelProfile, ParsedReply
from .registry import available_profiles, register, select_profile

for _module in pkgutil.iter_modules(__path__):
    if _module.name not in ("base", "registry"):
        importlib.import_module(f"{__name__}.{_module.name}")

__all__ = ["ModelProfile", "ParsedReply", "available_profiles", "register", "select_profile"]
