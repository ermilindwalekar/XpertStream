@echo off
:::
::: ExportNPUModel — Intel NPU Static-Shape Model Exporter & Quantizer
::: ===================================================================
:::
::: EXPORT QWEN 2.5 CODER 7B (Recommended for Panther Lake PTL / Lunar Lake):
:::   ExportNPUModel.cmd --coder
:::   ExportNPUModel.cmd --preset coder-7b
:::
::: LIGHTWEIGHT CODING / METEOR LAKE PRESET (1.5B):
:::   ExportNPUModel.cmd --preset coder-1.5b
:::
::: CUSTOM MODEL (Hugging Face Model ID or local path):
:::   ExportNPUModel.cmd --model-id "Qwen/Qwen2.5-Coder-7B-Instruct"
:::   ExportNPUModel.cmd --model-id "meta-llama/Meta-Llama-3.1-8B-Instruct"
:::
::: HUGGING FACE AUTHENTICATION (For gated models e.g. Llama-3.1, Llama-3.2):
:::   ExportNPUModel.cmd --login
:::   ExportNPUModel.cmd --token "hf_YourTokenHere"
:::
::: PRECISION (int4 [recommended for NPU memory bandwidth], int8, fp16):
:::   ExportNPUModel.cmd --coder --precision int4
:::
::: CONTEXT SEQUENCE LENGTH (Compile-time static hard bound):
:::   ExportNPUModel.cmd --coder --max-seq-len 2048
:::
::: REQUIREMENTS:
:::   - Python 3.10+ (python.org or Anaconda)
:::   - Dedicated virtual environment (.venv-npu) is created automatically
:::
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "RUN_ARGS=%*"

set "DO_LOGIN=0"
set "USER_TOKEN="
set "EXPECT_TOKEN=0"
for %%A in (!RUN_ARGS!) do (
  set "ARG=%%~A"
  if "!EXPECT_TOKEN!"=="1" (
    set "USER_TOKEN=!ARG!"
    set "EXPECT_TOKEN=0"
  ) else (
    if /I "!ARG!"=="--token" set "EXPECT_TOKEN=1"
    if /I "!ARG!"=="--hf-token" set "EXPECT_TOKEN=1"
    if /I "!ARG:~0,8!"=="--token=" set "USER_TOKEN=!ARG:~8!"
    if /I "!ARG:~0,11!"=="--hf-token=" set "USER_TOKEN=!ARG:~11!"
    if /I "!ARG!"=="--login" set "DO_LOGIN=1"
  )
)

echo ===================================================================
echo   XpertStream: Intel NPU Static-Shape Model Exporter Launcher
echo ===================================================================

rem 1. Check for Python launcher or python executable
where py >nul 2>&1
if not errorlevel 1 (
  set "PYTHON_LAUNCHER=py -3"
) else (
  where python >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_LAUNCHER=python"
  ) else (
    echo [ExportNPUModel] [ERROR] Python was not found on PATH.
    echo [ExportNPUModel] Please install Python 3.10+ from https://python.org or Anaconda and add it to PATH.
    pause
    exit /b 1
  )
)

rem 2. Ensure dedicated virtual environment (.venv-npu) exists
if not exist ".venv-npu\Scripts\python.exe" (
  echo [ExportNPUModel] Virtual environment '.venv-npu' not found. Creating one now...
  %PYTHON_LAUNCHER% -m venv .venv-npu
  if errorlevel 1 (
    echo [ExportNPUModel] [ERROR] Failed to create virtual environment '.venv-npu'.
    pause
    exit /b 1
  )
  echo [ExportNPUModel] Virtual environment created successfully.
)

rem 3. Activate virtual environment
call .venv-npu\Scripts\activate.bat
if errorlevel 1 (
  echo [ExportNPUModel] [ERROR] Failed to activate virtual environment.
  pause
  exit /b 1
)

rem 4. Verify required packages are installed
echo [ExportNPUModel] Verifying Python dependencies...
python -c "import openvino, openvino_genai, optimum.intel.openvino, nncf, transformers" >nul 2>&1
if errorlevel 1 (
  echo [ExportNPUModel] Required packages not found. Installing dependencies from requirements-npu.txt...
  python -m pip install --upgrade pip
  python -m pip install -r requirements-npu.txt
  if errorlevel 1 (
    echo [ExportNPUModel] [ERROR] Failed to install dependencies.
    pause
    exit /b 1
  )
  echo [ExportNPUModel] Dependencies installed successfully.
) else (
  echo [ExportNPUModel] Dependencies verified.
)

rem 5. Handle --login mode
if "!DO_LOGIN!"=="1" (
  echo [ExportNPUModel] ---------------------------------------------------
  echo [ExportNPUModel] Hugging Face Access Token Setup
  echo [ExportNPUModel] ---------------------------------------------------
  if "!USER_TOKEN!"=="" (
    set /p "USER_TOKEN=Paste your Hugging Face Token (hf_...): "
  )
  if not "!USER_TOKEN!"=="" (
    set "HF_TOKEN=!USER_TOKEN!"
    python -c "import os, huggingface_hub; huggingface_hub.login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)"
    if not errorlevel 1 (
      echo.
      echo [ExportNPUModel] [SUCCESS] Token saved to Hugging Face credential cache!
      echo [ExportNPUModel] You can now run model exports without re-entering the token.
    ) else (
      echo.
      echo [ExportNPUModel] [ERROR] Failed to authenticate token. Please check your token and try again.
    )
  ) else (
    echo.
    echo [ExportNPUModel] [ERROR] No token provided.
  )
  echo ===================================================================
  pause
  exit /b 0
)

rem 6. Detect or forward token to Python environment
if defined HF_TOKEN (
  echo [ExportNPUModel] Detected HF_TOKEN in environment.
) else if not "!USER_TOKEN!"=="" (
  echo [ExportNPUModel] Using provided Hugging Face token.
  set "HF_TOKEN=!USER_TOKEN!"
)

rem 7. Run the model export script
echo [ExportNPUModel] Executing model export with arguments:
if "%RUN_ARGS%"=="" (
  echo [ExportNPUModel]   (Using default model and settings)
) else (
  if defined USER_TOKEN (
    echo [ExportNPUModel]   Arguments include a token - hidden
  ) else (
    echo [ExportNPUModel]   !RUN_ARGS!
  )
)
echo.

python scripts\export_npu_model.py !RUN_ARGS!
set EXIT_CODE=%errorlevel%

echo.
if not "%EXIT_CODE%"=="0" (
  echo [ExportNPUModel] [ERROR] Export script exited with failure code %EXIT_CODE%.
) else (
  echo [ExportNPUModel] [SUCCESS] Model export and static shape optimization complete!
)

echo ===================================================================
pause
exit /b %EXIT_CODE%
