@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" goto :setup_required
if not exist "venv\Scripts\pythonw.exe" goto :setup_required

venv\Scripts\python.exe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('PyQt5') else 1)" >nul 2>&1
if errorlevel 1 goto :setup_required

start "" "venv\Scripts\pythonw.exe" -m tools.traceiso_launcher_gui
exit /b 0

:setup_required
echo TraceISO is not set up yet, or its GUI dependency is missing.
echo Run 00_Setup_Update_Dependencies.bat, then launch TraceISO again.
echo.
pause
exit /b 1
