@echo off
setlocal
REM =====================================================================
REM  Run Remove-AI-Watermarks GUI
REM  Double-click this file to launch the Qt6 desktop interface.
REM  Requires: uv (https://docs.astral.sh/uv/)
REM =====================================================================
chcp 65001 >nul

cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv tidak ditemukan. Install dulu: https://docs.astral.sh/uv/#installation
    pause
    exit /b 1
)

REM Ensure GUI dependencies are installed (idempotent, fast if already present)
echo Checking GUI dependencies...
uv run --extra gui --quiet python -c "import PyQt6" >nul 2>nul
if errorlevel 1 (
    echo Installing GUI dependencies ^(PyQt6^)...
    uv sync --extra gui
    if errorlevel 1 (
        echo [ERROR] Gagal menginstal dependency GUI.
        pause
        exit /b 1
    )
)

echo.
echo Starting Remove-AI-Watermarks GUI...
uv run remove-ai-watermarks-gui
if errorlevel 1 (
    echo [ERROR] GUI exited with an error ^(code %errorlevel%^).
    pause
    exit /b 1
)

endlocal
