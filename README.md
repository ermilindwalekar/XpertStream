# XpertStream

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Target: Intel NPU](https://img.shields.io/badge/Target-Intel%20NPU%20%7C%20GPU%20%7C%20CPU-orange.svg)](https://www.intel.com/)

**XpertStream** exports local models to OpenVINO IR and runs them through OpenVINO GenAI on Intel NPU, Intel GPU, or CPU. [OpenCode](https://opencode.ai/docs/) provides the coding-agent interface, including project tools and file editing; XpertStream provides model inference through a local API.

```mermaid
graph LR
    HF[Hugging Face model] -->|ExportNPUModel.cmd: INT4 + OpenVINO IR| Models[models/]
    Models --> Server[AgentServer.cmd: OpenAI-compatible server]
    Server -->|localhost /v1/chat/completions| OpenCode[OpenCode agent]
    OpenCode --> Project[Your project]
```

---

## Set Up on Another Windows Machine

Clone the repository and install Python 3.10+ (with the `py` launcher or `python` on PATH), Git, and the Intel GPU and/or NPU drivers for the target hardware. Model size, available memory, and device limits determine which exported models the hardware can run. The Windows `.cmd` launchers create and manage the local `.venv-npu` Python environment; install device drivers and the OpenCode CLI separately.

The first `ExportNPUModel.cmd --preset <name>` creates the environment, installs [requirements-npu.txt](requirements-npu.txt) from PyPI, downloads the checkpoint from Hugging Face, and writes an OpenVINO model under `models/`. Provide internet access and sufficient free disk space for the checkpoint and exported files. For gated models, use `ExportNPUModel.cmd --login` and enter your token at the prompt; Qwen3-30B-A3B is publicly downloadable. Each machine maintains its own `.venv-npu/`, Hugging Face cache, and exported models because these machine-specific artifacts are excluded from Git. After cloning, export the model on that machine or copy a compatible exported model directory; create the virtual environment locally rather than copying it between machines.

For a 30B export, reserve substantially more disk and RAM/pagefile space than the final INT4 model size. On a machine with roughly 63 GB RAM and a 57 GB pagefile, the conversion process crashed in PyTorch after loading the checkpoint. The current validation covers the export preset and command construction; complete conversion and iGPU inference remain to be tested on a higher-memory system.

After export, use [Chat.cmd](Chat.cmd) for console chat or [AgentServer.cmd](AgentServer.cmd) to serve the model through a local API. Install the [OpenCode CLI](https://opencode.ai/docs/#install) for coding-agent workflows. VS Code and the optional OpenCode extension provide the editor-integrated terminal experience.

## Running a Real LLM on Intel NPU

To run an exported LLM on a supported Intel NPU, export it to OpenVINO IR with weight compression, then compile and execute it through OpenVINO GenAI:

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
- Writes `npu_static_config.json` with model metadata and the requested sequence-length budget. Console chat reads this metadata; AgentServer takes its prompt and response budgets from command-line options.

### Step 2: Interactive Terminal Chat Mode
Launch the multi-turn interactive chat terminal [Chat.cmd](Chat.cmd):

```bat
:: Launch interactive chat session (prefers GPU, then NPU, then CPU)
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
- **Multi-turn Context:** Includes prior user and assistant messages in each request until `/reset`.
- **Context Monitoring:** `/tokens` reports the conversation token count against the sequence-length value in the export metadata; chat warns near that value.
- **Context Reset:** `/reset` clears the client-side conversation history.
- **Custom System Prompt:** `/system <instructions>` updates persona dynamically.

### Step 3: Single-Shot Inference Script
Run one-off prompts with [RunNPU.cmd](RunNPU.cmd):

```bat
RunNPU.cmd "Explain quantum computing in one sentence"
```

### Step 4: OpenCode inside VS Code
Use OpenCode to work on project files; the XpertStream server provides local model inference. Install the [OpenCode CLI](https://opencode.ai/docs/#install) separately on Windows (the standalone `opencode.exe` can be placed in `%USERPROFILE%\.local\bin`). The optional `sst-dev.opencode` extension provides VS Code terminal integration. No Copilot subscription is needed.

- [scripts/agent_server.py](scripts/agent_server.py) exposes `/v1/chat/completions`, `/v1/models`, and `/health` on localhost. OpenCode sends requests to this local OpenAI-compatible API. Model profiles apply model-family chat formatting, generation defaults, and tool-call parsing. `--log-output` prints raw model responses for debugging.
- `AgentServer.cmd` registers the active model in OpenCode's user config after the server loads when the config is absent or already XpertStream-only. It leaves unrelated user configurations unchanged. [OpenCode.cmd](OpenCode.cmd) launches the terminal client using an isolated config stored with XpertStream; OpenCode operates in the shell's current project directory. Both entry points connect to the same local model server.
- [demo-project/](demo-project/) is a small Python project for checking the integration; its README defines the test command.

```bat
:: Terminal 1: start one OpenVINO server and leave it running.
cd /d C:\XpertStream
AgentServer.cmd --device GPU.0 --model qwen2.5-coder-7b-npu --max-prompt-len 16384
:: Or try --device NPU with a context size your hardware can actually handle.

:: In VS Code: File > Open Folder... > C:\XpertStream\demo-project
:: Open OpenCode from the VS Code Command Palette or its terminal integration.
:: In OpenCode, /models lists the local xpertstream/<loaded-model> entry.
```

The [OpenCode VS Code integration](https://opencode.ai/docs/ide/) opens the agent in a dedicated terminal panel. Reload VS Code after installing the extension, then choose **Open opencode** from the Command Palette. Start a new OpenCode session after changing server models to load the updated model selection. The configurator reads `/health` and `/v1/models` from the running server before writing the local provider config. Set `XPERT_ENDPOINT` when using a non-default server port. `AgentServer.cmd --device` selects the inference device; initialization errors are reported by the server.

In a read-only smoke test, OpenCode's instructions and tools used about 7.9K prompt tokens. Configure the server prompt budget to accommodate the agent prompt and the target device's memory capacity; the server reports requests that exceed its configured budget. OpenCode manages file edits and approval policies. Evaluate tool-call parsing and coding-task reliability as separate model capabilities, and verify changes with test output before using a model as a research baseline.

The inference boundary is [scripts/inference_backend.py](scripts/inference_backend.py). It constructs the OpenVINO pipeline and configures NPU budgets/cache or iGPU execution. This boundary isolates model execution from the agent and provides the integration point for future expert selection, pruning, loading, and residency research. The current pipeline loads the exported model as a whole; expert-level residency requires additional runtime support.

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
The Hugging Face model card describes Qwen3-8B as Apache-2.0 and supports agent/tool calls. Its preset writes the OpenVINO export to `models/qwen3-8b-npu`, alongside the existing Qwen and Llama exports:

```bat
ExportNPUModel.cmd --preset qwen3-8b --max-seq-len 16384 --precision int4
```

Then run it on the Intel iGPU or NPU using `AgentServer.cmd --device GPU.0 --model qwen3-8b-npu --max-prompt-len 16384` (select `NPU` when testing on PTL). The Qwen3-30B-A3B MoE below is a separate, larger model. The current implementation loads each exported model as a whole; expert paging and pruning are future runtime research areas.

### Qwen3-30B-A3B MoE GPU trial
The Apache-2.0 [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) model has 30.5B total and 3.3B active parameters. Export it to its own directory, then select the Intel iGPU explicitly:

```bat
ExportNPUModel.cmd --preset qwen3-30b-a3b --precision int4 --max-seq-len 16384
Chat.cmd --device GPU --model qwen3-30b-a3b-npu
:: Or serve it to OpenCode from a separate terminal:
AgentServer.cmd --device GPU.0 --model qwen3-30b-a3b-npu --max-prompt-len 16384
```

The exporter's `-npu` directory suffix identifies the model-output convention; the `--device` option independently selects the runtime device. The existing `qwen3` model profile covers `qwen3_moe` on the server. The current runtime loads the full quantized checkpoint, so memory planning must account for all 30.5B parameters, along with runtime buffers and context. Conversion and iGPU loading of this model remain to be validated; available memory and maximum allocation size determine whether a particular iGPU can load it. `Chat.cmd` reports the selected device and falls back to CPU when GPU initialization fails. `AgentServer.cmd` reports initialization errors directly.

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
