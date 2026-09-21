@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" goto :setup_required
if not exist "venv\Scripts\pythonw.exe" goto :setup_required

"venv\Scripts\python.exe" -c "from PyQt5 import QtCore, QtWidgets" >nul 2>&1
if errorlevel 1 goto :setup_required

start "" "venv\Scripts\pythonw.exe" -m tools.global_uncertainty_manager
exit /b 0

:setup_required
echo Global Uncertainty Manager is not set up yet, or its GUI dependency is unavailable.
echo Run 00_Setup_Update_Dependencies.bat, then launch it again.
echo.
pause
exit /b 1
