@echo off
setlocal
REM First-time setup and launch for Remove-AI-Watermarks GUI
REM Requires uv: https://docs.astral.sh/uv/
chcp 65001 >nul
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv tidak ditemukan. Install: https://docs.astral.sh/uv/#installation
    pause
    exit /b 1
)

echo Installing/synchronizing GUI dependencies...
uv sync --extra gui
if errorlevel 1 (
    echo [ERROR] Setup gagal.
    pause
    exit /b 1
)

echo Starting Remove-AI-Watermarks GUI...
uv run remove-ai-watermarks-gui
if errorlevel 1 (
    echo [ERROR] GUI exited with error code %errorlevel%.
    pause
    exit /b 1
)
endlocal
