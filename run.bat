@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
if "%~1"=="" (
  echo Usage: run notebooks\file.py
  exit /b 1
)
python %*
