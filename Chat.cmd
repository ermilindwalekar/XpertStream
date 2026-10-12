@echo off
:::
::: Chat — Interactive Multi-Turn Terminal Chat Session
::: ====================================================
:::
::: QUICK START:
:::   Chat.cmd
:::
::: HARDWARE SELECTION:
:::   Chat.cmd --device GPU   (Intel Arc / Xe Graphics - Recommended fast response)
:::   Chat.cmd --device NPU   (Intel AI Boost NPU)
:::   Chat.cmd --device CPU   (Host CPU)
:::
::: CUSTOM PARAMETERS:
:::   Chat.cmd --temperature 0.7 --max-tokens 512
:::
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

if not exist ".venv-npu\Scripts\python.exe" (
  echo [Chat] [ERROR] Virtual environment '.venv-npu' not found.
  echo [Chat] Run ExportNPUModel.cmd first to initialize dependencies and export a model.
  pause
  exit /b 1
)

call .venv-npu\Scripts\activate.bat

python -c "import openvino_genai" >nul 2>&1
if errorlevel 1 (
  echo [Chat] Required 'openvino-genai' runtime not found. Installing from requirements-npu.txt...
  python -m pip install -r requirements-npu.txt
)

python scripts\chat_terminal.py %*
set EXIT_CODE=%errorlevel%

exit /b %EXIT_CODE%
