"""
XpertStream: Interactive Multi-Turn Terminal Chat
=================================================
Interactive, streaming terminal chat session with multi-turn conversation memory,
hardware selection (NPU, GPU, CPU), dynamic context length tracking, and agent commands.
"""

import sys
import os
import time
import argparse
from pathlib import Path

def print_banner(device: str, model_id: str, max_seq_len: int):
    print("\n" + "=" * 70)
    print("           XpertStream: Interactive Terminal Chat Session             ")
    print("=" * 70)
    print(f"[*] Hardware Accelerator:  {device}")
    print(f"[*] Loaded Model:          {model_id}")
    print(f"[*] Static Max Context:    {max_seq_len} tokens")
    print(f"[*] Special Commands:")
    print(f"    - '/reset' or '/clear' : Reset conversation history & KV-cache")
    print(f"    - '/tokens'            : Show current context token usage")
    print(f"    - '/system <prompt>'   : Change system instructions")
    print(f"    - 'exit' or 'quit'     : End session")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="XpertStream Interactive Terminal Chat")
    parser.add_argument("--device", "-d", choices=["AUTO", "NPU", "GPU", "CPU"], default="AUTO", help="Target accelerator")
    parser.add_argument("--model", type=str, default=None, help="Model preset or folder name (e.g. coder-7b, qwen2.5-coder-7b-npu)")
    parser.add_argument("--coder", action="store_true", help="Load exported Qwen2.5-Coder-7B model")
    parser.add_argument("--model-dir", "-m", default=None, help="Path to exported OpenVINO model")
    parser.add_argument("--max-tokens", "-n", type=int, default=512, help="Max new tokens per response")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    # Determine model directory
    model_path = None
    if args.coder:
        model_path = Path("./models/qwen2.5-coder-7b-npu").resolve()
    elif args.model:
        if (Path(args.model) / "openvino_model.xml").exists():
            model_path = Path(args.model).resolve()
        elif (Path("./models") / args.model / "openvino_model.xml").exists():
            model_path = (Path("./models") / args.model).resolve()
        elif (Path("./models") / f"{args.model}-npu" / "openvino_model.xml").exists():
            model_path = (Path("./models") / f"{args.model}-npu").resolve()
        elif args.model in ["coder-7b", "7b", "coder"]:
            model_path = Path("./models/qwen2.5-coder-7b-npu").resolve()
    elif args.model_dir:
        model_path = Path(args.model_dir).resolve()

    if not model_path or not (model_path / "openvino_model.xml").exists():
        # Check standard locations with priority: coder-7b -> latest modified model -> qwen2.5-npu
        models_root = Path("./models").resolve()
        preferred_coder = models_root / "qwen2.5-coder-7b-npu"
        if (preferred_coder / "openvino_model.xml").exists():
            model_path = preferred_coder
            print(f"[*] Detected Qwen 2.5 Coder 7B model at: {model_path}")
        else:
            candidates = list(models_root.glob("*/openvino_model.xml")) if models_root.exists() else []
            if candidates:
                # Sort by modification time, most recent first
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_path = candidates[0].parent
                print(f"[*] Auto-detected exported model at: {model_path}")
            else:
                print(f"[ERROR] No exported OpenVINO model found in ./models/")
                print("[!] Run ExportNPUModel.cmd first to export a model.")
                sys.exit(1)

    import openvino as ov
    import openvino_genai as og
    from transformers import AutoTokenizer

    core = ov.Core()
    available_devices = core.available_devices

    # Device selection
    target_device = args.device
    if target_device == "AUTO":
        if "GPU" in available_devices or "GPU.0" in available_devices:
            target_device = "GPU"
        elif "NPU" in available_devices:
            target_device = "NPU"
        else:
            target_device = "CPU"

    try:
        full_dev_name = core.get_property(target_device, "FULL_DEVICE_NAME")
    except Exception:
        full_dev_name = target_device

    print(f"[*] Loading tokenizer and model onto {target_device} ({full_dev_name})...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Read config metadata if available
    max_seq_len = 2048
    config_file = model_path / "npu_static_config.json"
    if config_file.exists():
        import json
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            max_seq_len = cfg.get("max_seq_len", 2048)

    t0 = time.perf_counter()
    pipe = None
    try:
        if target_device == "NPU":
            cache_dir = model_path / "cache_npu"
            cache_dir.mkdir(exist_ok=True)
            pipe = og.LLMPipeline(str(model_path), "NPU", CACHE_DIR=str(cache_dir))
        else:
            pipe = og.LLMPipeline(str(model_path), target_device)
        t_load = time.perf_counter() - t0
        print(f"[+] Initialized {target_device} pipeline in {t_load:.2f}s!")
    except Exception as e:
        print(f"[!] Initialization on {target_device} failed: {e}")
        fallback = "CPU"
        print(f"[*] Falling back to {fallback}...")
        pipe = og.LLMPipeline(str(model_path), fallback)
        target_device = fallback

    print_banner(f"{target_device} ({full_dev_name})", model_path.name, max_seq_len)

    # Conversation state
    system_prompt = "You are a helpful, intelligent, and concise AI assistant."
    history = []

    gen_config = og.GenerationConfig()
    gen_config.max_new_tokens = args.max_tokens
    gen_config.temperature = args.temperature

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Session ended. Goodbye!")
            break

        if user_input.lower() in ["/reset", "/clear"]:
            history.clear()
            print("[*] Conversation history and context memory reset.")
            continue

        if user_input.lower().startswith("/system"):
            parts = user_input.split(" ", 1)
            if len(parts) > 1 and parts[1].strip():
                system_prompt = parts[1].strip()
                print(f"[*] System prompt updated to: \"{system_prompt}\"")
            else:
                print(f"[*] Current system prompt: \"{system_prompt}\"")
            continue

        if user_input.lower() == "/tokens":
            # Estimate token usage
            messages = [{"role": "system", "content": system_prompt}] + history
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens_count = len(tokenizer.encode(prompt_str))
            print(f"[*] Context tokens used: {tokens_count} / {max_seq_len} ({max_seq_len - tokens_count} remaining)")
            continue

        # Add user message
        history.append({"role": "user", "content": user_input})

        # Build full conversation prompt using ChatML / model chat template
        messages = [{"role": "system", "content": system_prompt}] + history
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        current_tokens = len(tokenizer.encode(prompt_text))

        # Check hard boundary
        if current_tokens >= max_seq_len - 64:
            print(f"\n[XpertStream::WARNING] Sequence context limit approaching ({current_tokens}/{max_seq_len} tokens)!")
            print("[!] Consider typing '/reset' to start a fresh context session.")

        print("\nAssistant > ", end="", flush=True)

        full_reply = []
        def streamer(subword: str) -> bool:
            print(subword, end="", flush=True)
            full_reply.append(subword)
            return False  # Continue streaming

        t_gen_start = time.perf_counter()
        try:
            pipe.generate(prompt_text, gen_config, streamer)
        except Exception as e:
            print(f"\n[!] Inference runtime error: {e}")
            history.pop()  # Remove last user message on failure
            continue
        t_gen = time.perf_counter() - t_gen_start

        assistant_text = "".join(full_reply).strip()
        history.append({"role": "assistant", "content": assistant_text})

        # Calculate metrics
        reply_token_count = len(tokenizer.encode(assistant_text))
        tps = reply_token_count / t_gen if t_gen > 0 else 0
        print(f"\n[{reply_token_count} tokens | {t_gen:.2f}s | {tps:.1f} tok/s]")

if __name__ == "__main__":
    main()
