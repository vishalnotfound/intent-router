@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

if not exist ".venv\Scripts\python.exe" (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install Python 3.10 or newer and try again.
        exit /b 1
    )
)

"%PYTHON%" -c "import uvicorn" >nul 2>nul
if errorlevel 1 (
    echo Dependencies are not installed. Follow the Windows setup instructions in README.md.
    exit /b 1
)

"%PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
if errorlevel 1 (
    echo The server stopped with an error.
    exit /b 1
)
