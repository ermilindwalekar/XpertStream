"""
XpertStream: Live Intel NPU / GPU / CPU Local LLM Pipeline
===========================================================
Runs stateful LLMs locally on Intel hardware:
- NPU (Intel(R) AI Boost / NPU 3720 / 40xx)
- GPU (Intel Arc / Xe Graphics)
- CPU
"""

import sys
import os
import time
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="XpertStream Local Intel LLM Inference")
    parser.add_argument("prompt", nargs="*", default=["What is 15 multiplied by 42? Explain simply."], help="Prompt text")
    parser.add_argument("--device", "-d", choices=["NPU", "GPU", "CPU", "AUTO"], default="AUTO", help="Inference device (default: AUTO - prefers NPU, then GPU, then CPU)")
    parser.add_argument("--model", type=str, default=None, help="Model preset or folder name (e.g. coder-7b, qwen2.5-coder-7b-npu)")
    parser.add_argument("--coder", action="store_true", help="Load exported Qwen2.5-Coder-7B model")
    parser.add_argument("--model-dir", "-m", default=None, help="Path to exported OpenVINO model directory")
    parser.add_argument("--max-tokens", "-n", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", "-t", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)")
    args = parser.parse_args()

    user_prompt = " ".join(args.prompt) if isinstance(args.prompt, list) else str(args.prompt)

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
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_path = candidates[0].parent
                print(f"[*] Auto-detected exported model at: {model_path}")
            else:
                print(f"[ERROR] No exported OpenVINO model found in ./models/")
                print("[!] Run ExportNPUModel.cmd first to export a model.")
                sys.exit(1)

    import openvino as ov
    import openvino_genai as og

    core = ov.Core()
    available_devices = core.available_devices

    # Determine target device
    target_device = args.device
    if target_device == "AUTO":
        if "NPU" in available_devices:
            target_device = "NPU"
        elif "GPU" in available_devices or "GPU.0" in available_devices:
            target_device = "GPU"
        else:
            target_device = "CPU"

    # Resolve device name
    try:
        full_dev_name = core.get_property(target_device, "FULL_DEVICE_NAME")
    except Exception:
        full_dev_name = target_device

    print("===================================================================")
    print("               XpertStream Local LLM Inference Engine              ")
    print("===================================================================")
    print(f"[*] Available Hardware: {available_devices}")
    print(f"[*] Selected Device:    {target_device} ({full_dev_name})")
    print(f"[*] Model Directory:    {model_path}")
    print(f"[*] Prompt:             \"{user_prompt}\"")
    print("-------------------------------------------------------------------")
    print(f"[*] Loading model onto {target_device}...")

    t0 = time.perf_counter()
    pipe = None

    # Load onto target device with graceful fallback if NPU driver compiler times out or resets
    def create_pipeline(device):
        if device == "NPU":
            cache_dir = model_path / "cache_npu"
            cache_dir.mkdir(exist_ok=True)
            return og.LLMPipeline(str(model_path), "NPU", CACHE_DIR=str(cache_dir))
        return og.LLMPipeline(str(model_path), device)

    try:
        pipe = create_pipeline(target_device)
        t_load = time.perf_counter() - t0
        print(f"[+] Successfully initialized on {target_device} in {t_load:.2f}s!")
    except Exception as e:
        print(f"[!] Initialization on {target_device} encountered: {e}")
        fallback_device = "GPU" if ("GPU" in available_devices or "GPU.0" in available_devices) else "CPU"
        print(f"[*] Falling back to {fallback_device}...")
        t0 = time.perf_counter()
        pipe = create_pipeline(fallback_device)
        t_load = time.perf_counter() - t0
        target_device = fallback_device
        print(f"[+] Successfully initialized on {target_device} in {t_load:.2f}s!")

    print("\n-------------------------------------------------------------------")
    print(f"[*] Streaming response ({target_device}):")
    print("-------------------------------------------------------------------")

    def streamer(token: str) -> bool:
        print(token, end="", flush=True)
        return False  # Return False to continue streaming

    gen_config = og.GenerationConfig()
    gen_config.max_new_tokens = args.max_tokens
    gen_config.temperature = args.temperature

    t_infer_start = time.perf_counter()
    try:
        pipe.generate(user_prompt, gen_config, streamer)
    except Exception as e:
        print(f"\n[!] Hardware error on {target_device} during execution: {e}")
        if target_device == "NPU":
            fallback_device = "GPU" if ("GPU" in available_devices or "GPU.0" in available_devices) else "CPU"
            print(f"[*] Switching execution fallback to {fallback_device}...")
            pipe = create_pipeline(fallback_device)
            target_device = fallback_device
            print(f"[*] Streaming response ({target_device}):")
            pipe.generate(user_prompt, gen_config, streamer)
    t_infer = time.perf_counter() - t_infer_start

    print("\n-------------------------------------------------------------------")
    print(f"[+] Inference complete in {t_infer:.2f}s on {target_device}")
    print("===================================================================")

if __name__ == "__main__":
    main()
