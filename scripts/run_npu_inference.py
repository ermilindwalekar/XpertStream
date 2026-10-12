"""
XpertStream: Live Intel NPU Inference & ReAct Agent Pipeline
============================================================
Runs the exported static-shape model directly on the local Intel NPU ("Intel(R) AI Boost")
using OpenVINO with stateful ReadValue/Assign KV-Cache.
"""

import sys
import json
import time
from pathlib import Path
import openvino as ov
import openvino.properties.hint as hints
import numpy as np

def run_npu_inference(
    model_dir: str = "./models/qwen2.5-npu",
    user_prompt: str = "What is 15 multiplied by 42?",
    max_new_tokens: int = 128
):
    model_path = Path(model_dir).resolve()
    xml_path = model_path / "openvino_model.xml"

    if not xml_path.exists():
        print(f"[ERROR] Model file not found at: {xml_path}")
        sys.exit(1)

    print("===================================================================")
    print("       XpertStream: Live Intel NPU Stateful Inference Engine       ")
    print("===================================================================")

    # 1. Initialize OpenVINO Core & query Intel NPU
    core = ov.Core()
    devices = core.available_devices
    print(f"[*] Available devices: {devices}")

    target_device = "NPU" if "NPU" in devices else "CPU"
    device_name = core.get_property(target_device, "FULL_DEVICE_NAME")
    print(f"[*] Target device:     {target_device} ({device_name})")

    # 2. Load tokenizer
    from transformers import AutoTokenizer
    print(f"[*] Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # 3. Compile static shape model for NPU
    print(f"[*] Reading OpenVINO IR model: {xml_path.name}...")
    model = core.read_model(str(xml_path))

    # Set compile-time properties for Intel NPU
    npu_config = {
        hints.performance_mode(): hints.PerformanceMode.LATENCY,
    }

    print(f"[*] Compiling static model for {target_device}...")
    t_start_compile = time.perf_counter()
    compiled_model = core.compile_model(model, target_device, npu_config)
    t_compile = time.perf_counter() - t_start_compile
    print(f"[+] Compiled successfully in {t_compile:.2f}s!")

    # 4. Prepare OpenAI / ChatML formatted messages
    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
        {"role": "user", "content": user_prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\n-------------------------------------------------------------------")
    print("[*] Formatted Prompt:")
    print(prompt_text, end="")
    print("-------------------------------------------------------------------")

    input_tokens = tokenizer.encode(prompt_text, return_tensors="np")
    seq_len = input_tokens.shape[1]
    print(f"[*] Prompt Token Length: {seq_len} tokens")

    # 5. Stateful inference execution with ReadValue/Assign KV-Cache
    infer_request = compiled_model.create_infer_request()
    infer_request.reset_state()

    print("\n[*] Starting Autoregressive Token Generation on Intel NPU...")
    generated_tokens = []
    
    # Process prefill prompt
    t_gen_start = time.perf_counter()
    
    # We feed prompt tokens
    # Note: For stateful models, prefill passes prompt tokens, then decode loop runs step-by-step
    inputs = {}
    for input_port in compiled_model.inputs:
        name = input_port.get_any_name()
        if "input_ids" in name:
            inputs[input_port] = input_tokens
        elif "attention_mask" in name:
            inputs[input_port] = np.ones((1, seq_len), dtype=np.int64)
        elif "position_ids" in name:
            inputs[input_port] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
        elif "beam_idx" in name:
            inputs[input_port] = np.zeros((1,), dtype=np.int32)

    infer_request.start_async(inputs)
    infer_request.wait()
    
    # Get initial output logits
    logits = infer_request.get_output_tensor(0).data
    next_token = int(np.argmax(logits[0, -1, :]))
    generated_tokens.append(next_token)

    first_token_str = tokenizer.decode([next_token])
    print(first_token_str, end="", flush=True)

    # Autoregressive single-step decode loop
    current_pos = seq_len
    for step in range(max_new_tokens - 1):
        if next_token in [tokenizer.eos_token_id, 151645, 151643]:
            break

        current_pos += 1
        decode_input_ids = np.array([[next_token]], dtype=np.int64)
        decode_attention_mask = np.ones((1, current_pos), dtype=np.int64)
        decode_position_ids = np.array([[current_pos - 1]], dtype=np.int64)

        step_inputs = {}
        for input_port in compiled_model.inputs:
            name = input_port.get_any_name()
            if "input_ids" in name:
                step_inputs[input_port] = decode_input_ids
            elif "attention_mask" in name:
                step_inputs[input_port] = decode_attention_mask
            elif "position_ids" in name:
                step_inputs[input_port] = decode_position_ids
            elif "beam_idx" in name:
                step_inputs[input_port] = np.zeros((1,), dtype=np.int32)

        infer_request.start_async(step_inputs)
        infer_request.wait()

        logits = infer_request.get_output_tensor(0).data
        next_token = int(np.argmax(logits[0, -1, :]))
        generated_tokens.append(next_token)

        tok_str = tokenizer.decode([next_token])
        print(tok_str, end="", flush=True)

    t_total = time.perf_counter() - t_gen_start
    num_gen = len(generated_tokens)
    tps = num_gen / t_total if t_total > 0 else 0

    print("\n-------------------------------------------------------------------")
    print(f"[+] Generation complete: {num_gen} tokens in {t_total:.2f}s ({tps:.1f} tokens/sec)")
    print("===================================================================")

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is 15 multiplied by 42? Explain briefly."
    run_npu_inference(user_prompt=prompt)
