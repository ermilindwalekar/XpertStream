@echo off
:::
::: RunNPU — Execute Live LLM Inference & Agents on Local Intel Silicon
::: ====================================================================
:::
::: QUICK START:
:::   RunNPU.cmd
:::
::: PASS A CUSTOM PROMPT:
:::   RunNPU.cmd "Explain quantum computing in one sentence"
:::   RunNPU.cmd "Write a python function to compute fibonacci numbers"
:::
::: SELECT ACCELERATOR DEVICE:
:::   RunNPU.cmd --device NPU "What is 15 * 42?"   (Intel AI Boost NPU 3720 / 40xx)
:::   RunNPU.cmd --device GPU "What is 15 * 42?"   (Intel Arc / Xe Integrated Graphics)
:::   RunNPU.cmd --device CPU "What is 15 * 42?"   (Host CPU)
:::
::: MODEL DIRECTORY:
:::   Default: .\models\qwen2.5-npu
:::
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo ===================================================================
echo     XpertStream: Live Intel Stateful Inference Launcher
echo ===================================================================

if not exist ".venv-npu\Scripts\python.exe" (
  echo [RunNPU] [ERROR] Virtual environment '.venv-npu' not found.
  echo [RunNPU] Run ExportNPUModel.cmd first to set up the environment and export a model.
  pause
  exit /b 1
)

call .venv-npu\Scripts\activate.bat

python -c "import openvino_genai" >nul 2>&1
if errorlevel 1 (
  echo [RunNPU] Required 'openvino-genai' runtime not found. Installing from requirements-npu.txt...
  python -m pip install -r requirements-npu.txt
)

python scripts\run_npu_genai.py %*
set EXIT_CODE=%errorlevel%

echo.
pause
exit /b %EXIT_CODE%
