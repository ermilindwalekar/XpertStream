"""Write an isolated OpenCode config for the running local XpertStream model."""

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def fetch_model(endpoint: str) -> tuple[dict, str]:
    with urlopen(f"{endpoint}/health", timeout=5) as response:
        health = json.load(response)
    if health.get("status") != "ok":
        raise ValueError(f"XpertStream server is not healthy: {health.get('error')}")
    with urlopen(f"{endpoint}/v1/models", timeout=5) as response:
        model = json.load(response)["data"][0]["id"]
    return health, model


def make_config(current: dict, health: dict, model: str, endpoint: str) -> dict:
    updated = dict(current)
    provider = dict(updated.get("provider", {}))
    provider["xpertstream"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "XpertStream (local OpenVINO)",
        "options": {"baseURL": f"{endpoint}/v1", "apiKey": "local"},
        "models": {
            model: {
                "name": f"{model} ({health['device']})",
                "limit": {
                    "context": int(health["max_prompt_len"] * 0.85) + health["max_response_len"],
                    "output": health["max_response_len"],
                },
            }
        },
    }
    updated["provider"] = provider
    updated["model"] = f"xpertstream/{model}"
    updated["small_model"] = updated["model"]
    updated["default_agent"] = "xpert"
    updated["agent"] = {
        "xpert": {
            "description": "Focused local coding agent for XpertStream models",
            "mode": "primary",
            "model": f"xpertstream/{model}",
            "prompt": "You are a focused coding agent. Read project instructions before choosing commands. For test requests, follow the repository's documented test command, use the shell tool, and rely only on the actual exit code/output. Do not invent tool results or describe commands without executing them. Avoid unrelated tools and subagents.",
            "steps": 8,
            "permission": {
                "read": "allow",
                "glob": "allow",
                "grep": "allow",
                "list": "allow",
                "edit": "ask",
                "bash": "ask",
                "task": "deny",
                "webfetch": "deny",
                "websearch": "deny",
                "lsp": "deny",
                "skill": "deny",
                "question": "deny",
                "external_directory": "deny",
            },
        }
    }
    updated["share"] = "disabled"
    updated["enabled_providers"] = ["xpertstream"]
    return updated


def register_global_config(health: dict, model: str, endpoint: str, config: Path = None) -> bool:
    config = config or Path.home() / ".config" / "opencode" / "opencode.json"
    if config.with_suffix(".jsonc").exists():
        print(f"[OpenCode] Existing JSONC config at {config.with_suffix('.jsonc')}; not modifying user settings.")
        return False
    if config.exists():
        try:
            current = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"[OpenCode] Could not read {config}: {error}; not modifying it.")
            return False
        if current.get("enabled_providers") != ["xpertstream"]:
            print(f"[OpenCode] Existing config at {config} is not XpertStream-only; not modifying user settings.")
            return False
    else:
        current = {}
    updated = make_config(current, health, model, endpoint)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print(f"[OpenCode] Registered xpertstream/{model} for the VS Code OpenCode extension in {config}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / ".opencode-xpert.json")
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    try:
        health, model = fetch_model(endpoint)
    except (URLError, TimeoutError, ValueError, KeyError, IndexError) as error:
        parser.exit(1, f"[ERROR] Start a healthy AgentServer.cmd first: {error}\n")
    config = args.config
    updated = make_config({}, health, model, endpoint)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print(f"[+] OpenCode model {updated['model']} on {health['device']} in {config}")
    print(f"[+] Local endpoint: {endpoint}/v1")
    if health["max_prompt_len"] < 12000:
        print("[WARN] OpenCode's agent prompt may exceed this model's prompt budget. "
              "Try --max-prompt-len 16384 if the selected hardware supports it.")


if __name__ == "__main__":
    main()