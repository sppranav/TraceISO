@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Use the repository's public-index settings for this setup only.
rem This does not modify the user's global pip configuration.
set "PIP_CONFIG_FILE=%~dp0pip-public.ini"

echo ===================================================
echo      TraceISO - Setup / Dependency Update
echo ===================================================
echo.

set "PYTHON_CMD="
python --version >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=python"
if not defined PYTHON_CMD (
    py --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py"
)
if not defined PYTHON_CMD (
    echo Error: Python is not installed or is not available on PATH.
    echo Install Python 3.10+ and run this setup again.
    pause
    exit /b 1
)
%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Error: TraceISO requires Python 3.10 or newer.
    %PYTHON_CMD% --version
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [Setup] Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 goto :failed
)

echo [Setup] Updating pip...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo [Setup] Installing application dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [Setup] Installing GUI dependencies...
venv\Scripts\python.exe -m pip install --upgrade PyQt5 PyQt5-sip
if errorlevel 1 goto :failed

echo [Setup] Checking dependency compatibility...
venv\Scripts\python.exe -m pip check
if errorlevel 1 goto :failed

copy /y "requirements.txt" "venv\.requirements.snapshot" >nul

echo.
echo Setup completed successfully.
echo Normal launchers will now open without contacting package servers.
echo.
pause
exit /b 0

:failed
echo.
echo Setup failed. Review the messages above for the failing package or network request.
echo.
pause
exit /b 1
