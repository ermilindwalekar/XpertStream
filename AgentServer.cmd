@echo off
:::
::: AgentServer - Local OpenAI-compatible agent server for the @xpert VS Code chat participant
::: ==========================================================================================
:::
::: QUICK START:
:::   AgentServer.cmd                         (Qwen2.5-Coder-7B on NPU, http://127.0.0.1:8765)
:::
::: OPTIONS:
:::   AgentServer.cmd --device GPU
:::   AgentServer.cmd --max-prompt-len 8192 --max-response-len 1024
:::   AgentServer.cmd --model qwen2.5-npu --port 8765
:::
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

if not exist ".venv-npu\Scripts\python.exe" (
  echo [AgentServer] [ERROR] Virtual environment '.venv-npu' not found.
  echo [AgentServer] Run ExportNPUModel.cmd first to set up the environment and export a model.
  pause
  exit /b 1
)

call .venv-npu\Scripts\activate.bat

python -c "import openvino_genai, transformers" >nul 2>&1
if errorlevel 1 (
  echo [AgentServer] Required packages not found. Installing from requirements-npu.txt...
  python -m pip install -r requirements-npu.txt
)

python scripts\install_vscode_extension.py
if errorlevel 1 (
  echo [AgentServer] [WARN] Could not install the VS Code extension; the server will still start.
)

python scripts\agent_server.py %*
set EXIT_CODE=%errorlevel%

exit /b %EXIT_CODE%
