@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" goto :setup_required
if not exist "venv\Scripts\pythonw.exe" goto :setup_required

start "" "venv\Scripts\pythonw.exe" -m tools.desktop_startup crm
exit /b 0

:setup_required
echo CRM Library Manager is not set up yet, or its GUI dependency is missing.
echo Run 00_Setup_Update_Dependencies.bat, then launch it again.
echo.
pause
exit /b 1
