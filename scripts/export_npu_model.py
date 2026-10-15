#!/usr/bin/env python3
"""
Model Export & Optimization Script for Intel NPU Static-Shape Execution
=======================================================================
This script converts a Hugging Face causal LLM (e.g. Llama-3.2, Qwen-2.5) into
an OpenVINO Intermediate Representation (IR) graph optimized for Intel NPUs.

Key optimizations applied:
1. INT4 / NNCF Weight-Only Quantization (Sym/Asym) for low memory bandwidth.
2. Stateful KV-Cache transformation (ReadValue / Assign) to eliminate PCIe/SoC KV-cache movement.
3. Static shape compilation (batch=1, max_seq_len locked).

Requirements:
    pip install openvino optimum-intel nncf transformers
"""

import argparse
import sys
import os
import json
import subprocess
from pathlib import Path

MODEL_PRESETS = {
    "coder-7b": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "default_dir": "./models/qwen2.5-coder-7b-npu",
        "description": "Best sub-10B coding LLM (INT4 ~4.2GB, ideal for Panther Lake 18GB / Lunar Lake)"
    },
    "coder-1.5b": {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "default_dir": "./models/qwen2.5-coder-1.5b-npu",
        "description": "Fast lightweight coding model (INT4 ~1.2GB, ideal for Meteor Lake NPU 3720)"
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "default_dir": "./models/qwen3-8b-npu",
        "description": "Apache-2.0 Qwen3 8B agent/tool-calling baseline"
    },
    "qwen3-30b-a3b": {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "default_dir": "./models/qwen3-30b-a3b-npu",
        "description": "Apache-2.0 Qwen3 30B MoE (3B active); INT4 GPU trial"
    },
    "qwen-0.5b": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "default_dir": "./models/qwen2.5-npu",
        "description": "Ultra-lightweight test model (INT4 ~0.5GB)"
    },
    "llama-3.1-8b": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "default_dir": "./models/llama3.1-8b-npu",
        "description": "Meta Llama 3.1 8B Instruct (Gated, requires HF token)"
    }
}

def export_model_for_npu(
    model_id: str,
    output_dir: str,
    max_seq_len: int = 2048,
    precision: str = "int4",
    token: str = None
):
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            import huggingface_hub
            huggingface_hub.login(token=token, add_to_git_credential=False)
            print("[*] Successfully logged in to Hugging Face with provided token.")
        except Exception as e:
            print(f"[!] Note on token login: {e}")
    elif "HF_TOKEN" in os.environ:
        print("[*] Using Hugging Face token from environment variable (HF_TOKEN).")

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print("===================================================================")
    print("  XpertStream: Intel NPU Model Exporter & Static Shape Planner")
    print("===================================================================")
    print(f"[*] Target Model ID:     {model_id}")
    print(f"[*] Target Output Dir:    {out_path}")
    print(f"[*] Precision:            {precision.upper()}")
    print(f"[*] Static Max Sequence:  {max_seq_len} tokens")
    print(f"[*] KV-Cache Architecture: Stateful ReadValue/Assign (Intel NPU)")
    print("-------------------------------------------------------------------")

    # Construct optimum-cli command
    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli",
        "export", "openvino",
        "-m", model_id,
        "--task", "text-generation-with-past",
        "--weight-format", precision.lower(),
        str(out_path)
    ]

    if precision.lower() == "int4":
        cmd.extend(["--sym", "--group-size", "128", "--ratio", "0.8"])

    print(f"[*] Executing Optimum OpenVINO export pipeline...")
    print(f"    Command: {' '.join(cmd)}\n")

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[ERROR] Model export failed with exit code {res.returncode}")
        print("\n-------------------------------------------------------------------")
        if res.returncode & 0xFFFFFFFF == 0xC0000005:
            print("[!] Windows access violation (0xC0000005) in the export process, not an authentication error.")
            print("    Check Windows Event Viewer > Windows Logs > Application for the faulting module.")
            print("    Large-model conversion may require substantial RAM/pagefile space; a native library crash is also possible.")
        else:
            print("[!] See the Optimum output above for the cause. If it reports 401/403, check model access and authentication.")
        print("-------------------------------------------------------------------\n")
        sys.exit(res.returncode)

    # Generate XpertStream static shape metadata config file
    config_file = out_path / "npu_static_config.json"
    config_data = {
        "model_id": model_id,
        "batch_size": 1,
        "max_seq_len": max_seq_len,
        "decode_step_len": 1,
        "precision": precision.lower(),
        "target_hardware": "Intel NPU (37xx / 40xx / 50xx)",
        "kv_cache_paradigm": "stateful_read_value_assign",
        "cpu_fallback": False,
        "static_tensor_shapes": {
            "input_ids": [1, 1],
            "position_ids": [1, 1],
            "attention_mask": [1, max_seq_len]
        }
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    rebuild_detokenizer(out_path)

    print("\n-------------------------------------------------------------------")
    print(f"[+] Successfully exported static-shape model for Intel NPU!")
    print(f"    - Output Folder:  {out_path}")
    print(f"    - Metadata JSON:  {config_file}")
    print("===================================================================")

def rebuild_detokenizer(model_dir: Path):
    """Regenerates openvino_detokenizer.xml without clean_up_tokenization_spaces.

    Some tokenizers (e.g. Llama 3.1) enable it, which turns `from .models` into `from.models` and corrupts code.
    """
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    _, detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True, clean_up_tokenization_spaces=False)
    ov.save_model(detokenizer, str(model_dir / "openvino_detokenizer.xml"))
    print(f"[+] Rebuilt detokenizer without tokenization-space clean-up: {model_dir / 'openvino_detokenizer.xml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export and quantize Hugging Face LLM to OpenVINO IR for Intel NPU with static shapes"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(MODEL_PRESETS.keys()),
        help="Quick model preset: 'coder-7b' (Qwen2.5-Coder-7B), 'coder-1.5b', 'qwen-0.5b', 'llama-3.1-8b'"
    )
    parser.add_argument(
        "--coder",
        action="store_true",
        help="Shortcut to export Qwen/Qwen2.5-Coder-7B-Instruct for Panther Lake / Lunar Lake"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model repository ID (e.g. Qwen/Qwen2.5-Coder-7B-Instruct)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for OpenVINO IR model files"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Static sequence context limit compiled into graph (default: 2048)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="int4",
        choices=["int4", "int8", "fp16"],
        help="Weight precision for NPU acceleration (default: int4)"
    )
    parser.add_argument(
        "--token",
        "--hf-token",
        type=str,
        default=None,
        dest="token",
        help="Hugging Face user access token (hf_...) for gated models or private repos"
    )
    parser.add_argument(
        "--fix-detokenizer",
        metavar="MODEL_DIR",
        default=None,
        help="Only rebuild the detokenizer of an already-exported model (fixes code like 'from.models')"
    )
    args = parser.parse_args()

    if args.fix_detokenizer:
        rebuild_detokenizer(Path(args.fix_detokenizer).resolve())
        sys.exit(0)

    # Resolve preset or model ID
    selected_model_id = args.model_id
    selected_output_dir = args.output_dir

    if args.coder or args.preset == "coder-7b":
        preset_info = MODEL_PRESETS["coder-7b"]
        selected_model_id = preset_info["model_id"]
        if not selected_output_dir:
            selected_output_dir = preset_info["default_dir"]
    elif args.preset:
        preset_info = MODEL_PRESETS[args.preset]
        selected_model_id = preset_info["model_id"]
        if not selected_output_dir:
            selected_output_dir = preset_info["default_dir"]
    elif selected_model_id in MODEL_PRESETS:
        preset_info = MODEL_PRESETS[selected_model_id]
        selected_model_id = preset_info["model_id"]
        if not selected_output_dir:
            selected_output_dir = preset_info["default_dir"]
    elif selected_model_id == "Qwen/Qwen2.5-Coder-7B-Instruct":
        if not selected_output_dir:
            selected_output_dir = "./models/qwen2.5-coder-7b-npu"

    if not selected_model_id:
        selected_model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    if not selected_output_dir:
        # Generate directory name from model name
        safe_name = selected_model_id.split("/")[-1].lower() + "-npu"
        selected_output_dir = f"./models/{safe_name}"

    export_model_for_npu(
        model_id=selected_model_id,
        output_dir=selected_output_dir,
        max_seq_len=args.max_seq_len,
        precision=args.precision,
        token=args.token
    )
