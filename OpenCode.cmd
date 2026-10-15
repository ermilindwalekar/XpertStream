@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "XPERT_ROOT=%~dp0"
if not exist "%XPERT_ROOT%.venv-npu\Scripts\python.exe" (
  echo [OpenCode] [ERROR] Python environment missing. Run ExportNPUModel.cmd first.
  exit /b 1
)

if not defined XPERT_ENDPOINT set "XPERT_ENDPOINT=http://127.0.0.1:8765"
"%XPERT_ROOT%.venv-npu\Scripts\python.exe" "%XPERT_ROOT%scripts\configure_opencode.py" --endpoint "%XPERT_ENDPOINT%"
if errorlevel 1 exit /b 1

set "OPENCODE_CONFIG=%XPERT_ROOT%.opencode-xpert.json"
if exist "%USERPROFILE%\.local\bin\opencode.exe" (
  "%USERPROFILE%\.local\bin\opencode.exe" --pure %*
  exit /b !errorlevel!
)
for %%I in (opencode.exe) do if not "%%~$PATH:I"=="" (
  "%%~$PATH:I" --pure %*
  exit /b !errorlevel!
)

echo [OpenCode] [ERROR] OpenCode CLI not found. See https://opencode.ai/docs/ for installation.
exit /b 1