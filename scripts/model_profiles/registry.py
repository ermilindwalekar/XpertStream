"""Profile registration and selection."""

import json
from pathlib import Path

from .base import ModelProfile

_PROFILES: list = []


def register(cls):
    """Class decorator: makes a ModelProfile subclass available for auto-detection and --profile."""
    if not issubclass(cls, ModelProfile):
        raise TypeError(f"{cls.__name__} must subclass ModelProfile")
    if any(p.name == cls.name for p in _PROFILES):
        raise ValueError(f"Duplicate model profile name: {cls.name}")
    _PROFILES.append(cls)
    return cls


def available_profiles() -> dict:
    return {p.name: p for p in [*_PROFILES, ModelProfile]}


def select_profile(model_dir: Path, chat_template: str, override: str = None) -> ModelProfile:
    """Picks a profile by name, or by the model's config.json model_type and chat template."""
    profiles = available_profiles()
    if override:
        if override not in profiles:
            raise ValueError(f"Unknown profile {override!r}; available: {', '.join(profiles)}")
        return profiles[override]()
    config_file = Path(model_dir) / "config.json"
    model_type = ""
    if config_file.exists():
        model_type = json.loads(config_file.read_text(encoding="utf-8")).get("model_type", "")
    for cls in _PROFILES:
        if cls.matches(model_type, chat_template or ""):
            return cls()
    return ModelProfile()
