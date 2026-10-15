# XpertStream

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Target: Intel NPU](https://img.shields.io/badge/Target-Intel%20NPU%20%7C%20GPU%20%7C%20CPU-orange.svg)](https://www.intel.com/)

**XpertStream** serves local models with OpenVINO GenAI on Intel NPU or iGPU. Use [OpenCode](https://opencode.ai/docs/) as the separate open-source coding agent. XpertStream owns model export, inference and the local API; it does not own file editing or tool execution.

```mermaid
graph LR
    HF[Hugging Face model] -->|ExportNPUModel.cmd: INT4 + OpenVINO IR| Models[models/]
    Models --> Server[AgentServer.cmd: OpenAI-compatible server]
    Server -->|localhost /v1/chat/completions| OpenCode[OpenCode agent]
    OpenCode --> Project[Your project]
```

---

## Set Up on Another Windows Machine

Clone the repository and install Python 3.10+ (with the `py` launcher or `python` on PATH), Git, and the current Intel GPU and/or NPU drivers for the target hardware. A supported Intel device and enough memory are required to run its OpenVINO model; having a compatible driver does not guarantee a 30B model will load. The scripts use Windows `.cmd` launchers and create a local `.venv-npu` Python environment. They do not install hardware drivers or OpenCode.

The first `ExportNPUModel.cmd --preset <name>` creates the environment, installs [requirements-npu.txt](requirements-npu.txt) from PyPI, downloads the checkpoint from Hugging Face, and writes an OpenVINO model under `models/`. Internet access and ample free disk space are required. For gated models, use `ExportNPUModel.cmd --login` and enter your token at the prompt; Qwen3-30B-A3B is not gated. The `.venv-npu/`, Hugging Face download cache, and `models/` weights are **not in Git**: after a push and clone, export again or copy a compatible exported model directory separately. Do not copy the Python virtual environment between machines.

For a 30B export, plan for substantially more than the final INT4 model size in both disk and RAM/pagefile space. The attempt on a machine with roughly 63 GB RAM and a 57 GB pagefile crashed inside PyTorch after loading the checkpoint. A higher-memory machine is worth trying, but the OpenVINO export and iGPU runtime are still unverified for this model.

After export, use [Chat.cmd](Chat.cmd) for console chat or [AgentServer.cmd](AgentServer.cmd) for a local API. Coding-agent use additionally requires installing the [OpenCode CLI](https://opencode.ai/docs/#install) separately. VS Code and its optional OpenCode extension are only needed for the editor integration, not for export, console chat, or the terminal OpenCode client.

## Running a Real LLM on Intel NPU

To execute real LLM weights on an Intel NPU (Panther Lake PTL / Lunar Lake / Meteor Lake / Arrow Lake), the model undergoes static-shape compilation and INT4 quantization to match NPU hardware requirements:

### Step 1: Export & Quantize Model for Intel NPU
Use the Windows launcher [ExportNPUModel.cmd](ExportNPUModel.cmd):

```bat
:: Recommended Coding Model (Qwen 2.5 Coder 7B - ideal for Panther Lake 18GB / Lunar Lake)
ExportNPUModel.cmd --coder

:: Or using explicit preset:
ExportNPUModel.cmd --preset coder-7b

:: Lightweight coding model (Qwen 2.5 Coder 1.5B - ideal for Meteor Lake NPU 3720)
ExportNPUModel.cmd --preset coder-1.5b

:: One-time login with Hugging Face Token (for gated models like Llama-3.1 / 3.2)
ExportNPUModel.cmd --login
ExportNPUModel.cmd --model-id "meta-llama/Meta-Llama-3.1-8B-Instruct" --output-dir "./models/llama3.1-8b-npu"
```

The launcher:
- Auto-detects Python 3.10+ on the system.
- Creates an isolated `.venv-npu` virtual environment.
- Automatically installs/verifies [requirements-npu.txt](requirements-npu.txt) (`openvino`, `openvino-genai`, `optimum-intel`, `nncf`, `transformers`, `torch`).
- Exports a stateful OpenVINO model using Optimum's `text-generation-with-past` task.
- Compresses weights to INT4 by default; the compressed size depends on the model and mixed-precision ratio.
- Writes `npu_static_config.json` with the requested context budget; this metadata does not itself compile or enforce static tensor shapes.

### Step 2: Interactive Terminal Chat Mode
Launch the multi-turn interactive chat terminal [Chat.cmd](Chat.cmd):

```bat
:: Launch interactive chat session (auto-selects best local accelerator)
Chat.cmd

:: Run the exported Qwen 2.5 Coder 7B specifically on Intel NPU
Chat.cmd --device NPU --coder

:: Or select target silicon explicitly
Chat.cmd --device NPU     :: Intel AI Boost NPU (PTL / Lunar Lake / Meteor Lake)
Chat.cmd --device GPU     :: Intel Arc / Xe Integrated Graphics (fastest response)
Chat.cmd --device CPU     :: Host CPU
```

Features inside the chat session:
- **Streaming Tokens:** Real-time token output with generation throughput (`tokens/sec`).
- **Multi-turn Memory:** Keeps full conversational context across questions.
- **Context Monitoring:** `/tokens` checks used tokens vs. static hardware limit.
- **Context Reset:** `/reset` clears chat history and restarts fresh KV-cache.
- **Custom System Prompt:** `/system <instructions>` updates persona dynamically.

### Step 3: Single-Shot Inference Script
Run one-off prompts with [RunNPU.cmd](RunNPU.cmd):

```bat
RunNPU.cmd "Explain quantum computing in one sentence"
```

### Step 4: OpenCode inside VS Code
Use OpenCode to work on project files; the XpertStream server only handles inference. Install the [OpenCode CLI](https://opencode.ai/docs/#install) separately on Windows (the standalone `opencode.exe` can be placed in `%USERPROFILE%\.local\bin`). Install the optional `sst-dev.opencode` extension yourself if you want its VS Code terminal integration. No Copilot subscription is needed.

- [scripts/agent_server.py](scripts/agent_server.py) exposes `/v1/chat/completions`, `/v1/models`, and `/health` on localhost. OpenCode uses this API format without contacting OpenAI. Model profiles parse tool calls from each model family. `--log-output` prints raw responses for debugging.
- `AgentServer.cmd` registers the active model in OpenCode's user config when the server finishes loading. It will not overwrite a pre-existing non-XpertStream config. [OpenCode.cmd](OpenCode.cmd) remains a terminal alternative using an isolated ignored config; both paths point at the same local server.
- [demo-project/](demo-project/) is a small Python project for checking the integration; its README defines the test command.

```bat
:: Terminal 1: start one OpenVINO server and leave it running.
cd /d C:\XpertStream
AgentServer.cmd --device GPU.0 --model qwen2.5-coder-7b-npu --max-prompt-len 16384
:: Or try --device NPU with a context size your hardware can actually handle.

:: In VS Code: File > Open Folder... > C:\XpertStream\demo-project
:: Press Ctrl+Esc to open OpenCode in a VS Code split terminal (not Copilot Chat).
:: In OpenCode, /models lists the local xpertstream/<loaded-model> entry.
```

The [official OpenCode VS Code integration](https://opencode.ai/docs/ide/) is a dedicated split-terminal chat, not an entry in Copilot's Chat picker. Reload VS Code after installing the extension; open it using Command Palette > Open opencode if Ctrl+Esc is intercepted. Start a new OpenCode session after changing server models; the model menu shows the currently registered local model. The configurator uses the loaded model's `/health` and `/v1/models` results; `OpenCode.cmd` refuses an unhealthy or missing server. For a non-default port with the terminal launcher, set `XPERT_ENDPOINT` first. The server stays on the specified Intel device and never silently changes hardware.

OpenCode's own instructions and tools took about 7.9K prompt tokens in a read-only smoke test, so a 4K NPU prompt cannot run that agent unchanged. Start near 16K **only if** the target NPU or iGPU can support it; the server rejects over-budget requests. OpenCode handles its own edits and approval policies. A model that parses tool calls may still be too weak to complete coding tasks reliably. Verify changes with real test output before using it as a research baseline.

The inference boundary is [scripts/inference_backend.py](scripts/inference_backend.py). It constructs the OpenVINO pipeline and selects NPU static budgets/cache or iGPU execution. Future expert selection, pruning, loading and residency experiments belong behind this boundary, independent of the agent. This thin factory does **not** provide expert paging hooks: dynamic weight loading or expert pruning will require a separate inference backend or deeper model/runtime changes.

### Model profiles
Each model family writes tool calls differently. [scripts/model_profiles/](scripts/model_profiles/) holds one profile per family; the server picks one from the model's `config.json` (`model_type`) and chat template, and prints it at startup (`[*] Profile: ...`). Override with `--profile <name>`.

| Profile | Models | Tool-call format |
|---|---|---|
| `qwen2` | Qwen2 / Qwen2.5 (incl. Coder) | `<tool_call>{json}</tool_call>`; also accepts ```` ```json ```` blocks and bare JSON |
| `qwen3` | Qwen3 dense and MoE families | `<tool_call>{json}</tool_call>`; also accepts ```` ```json ```` blocks and bare JSON |
| `llama3` | Llama 3.x | whole reply is `{"name": ..., "parameters": {...}}` |
| `generic` | anything else | ```` ```json ```` blocks or bare JSON |

To add a model family, create `scripts/model_profiles/<family>.py` with a `ModelProfile` subclass decorated with `@register` (override `matches`, `parse`, and the streaming hooks as needed), and add tests with real captured outputs to [tests/test_model_profiles.py](tests/test_model_profiles.py). At startup the server also checks that the model's detokenizer doesn't alter code (e.g. `from .models` -> `from.models`); fix flagged models with `python scripts\export_npu_model.py --fix-detokenizer <model_dir>`.

### Qwen3-8B trial
The Hugging Face model card describes Qwen3-8B as Apache-2.0 and supports agent/tool calls. Export a separate OpenVINO copy without replacing the existing Qwen or Llama models:

```bat
ExportNPUModel.cmd --preset qwen3-8b --max-seq-len 16384 --precision int4
```

Then run it on the Intel iGPU or NPU using `AgentServer.cmd --device GPU.0 --model qwen3-8b-npu --max-prompt-len 16384` (replace `GPU.0` with `NPU` if testing PTL). The Qwen3-30B-A3B MoE below is a separate, larger model; the current backend does not dynamically page or prune experts.

### Qwen3-30B-A3B MoE GPU trial
The Apache-2.0 [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) (not Qwen3-Coder) has 30.5B total and 3.3B active parameters. Export to its own directory, then select the Intel iGPU explicitly:

```bat
ExportNPUModel.cmd --preset qwen3-30b-a3b --precision int4 --max-seq-len 16384
Chat.cmd --device GPU --model qwen3-30b-a3b-npu
:: Or serve it to OpenCode from a separate terminal:
AgentServer.cmd --device GPU.0 --model qwen3-30b-a3b-npu --max-prompt-len 16384
```

The `-npu` folder suffix is the exporter's existing naming convention; it does not force the GPU to run on an NPU. The existing `qwen3` model profile covers `qwen3_moe` on the server. This uses the full quantized model, not expert paging: all 30.5B parameters still need storage and enough runtime memory. Download/export requires substantial disk space and host RAM. Export and GPU compilation have not been validated for this 30B model; an iGPU may fail due to memory or per-allocation limits. `Chat.cmd` falls back to CPU if GPU initialization fails, so check its startup banner to confirm the actual device. `AgentServer.cmd` does not silently fall back.

```bat
:: Run the unit tests
.venv-npu\Scripts\python.exe -m unittest discover tests
```

---

## Repository Structure

```
c:\XpertStream\
├── ExportNPUModel.cmd            # Download from Hugging Face, INT4-quantize, export to OpenVINO IR
├── Chat.cmd                      # Interactive terminal chat
├── RunNPU.cmd                    # Single-shot prompt
├── AgentServer.cmd               # Local OpenVINO model server (NPU / iGPU)
├── OpenCode.cmd                  # Run external OpenCode agent against the loaded model
├── requirements-npu.txt
├── scripts/
│   ├── export_npu_model.py
│   ├── chat_terminal.py
│   ├── run_npu_genai.py
│   ├── run_npu_inference.py
│   ├── agent_server.py
│   ├── inference_backend.py      # OpenVINO runtime construction boundary
│   ├── configure_opencode.py     # Isolated local-only OpenCode provider configuration
│   ├── model_profiles/           # Per-model-family tool-call formats (qwen.py, llama.py, ...)
├── tests/                        # Unit tests (python -m unittest discover tests)
├── demo-project/                 # Small Python to-do app for trying the agent
└── models/                       # Exported models (gitignored)
```
