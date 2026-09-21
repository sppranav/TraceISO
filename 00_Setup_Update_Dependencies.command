#!/bin/bash
# TraceISO - Setup / Dependency Update (macOS)
#
# Double-click this file in Finder, or run it from Terminal.
# It is the macOS counterpart of 00_Setup_Update_Dependencies.bat.

cd "$(dirname "$0")" || exit 1

# Use the repository's public-index settings for this setup only.
# This does not modify the user's global pip configuration.
if [ -f "pip-public.ini" ]; then
    PIP_CONFIG_FILE="$PWD/pip-public.ini"
    export PIP_CONFIG_FILE
fi

echo "==================================================="
echo "     TraceISO - Setup / Dependency Update (macOS)"
echo "==================================================="
echo

fail() {
    echo
    echo "Setup failed. Review the messages above for the failing package"
    echo "or network request."
    echo
    read -r -p "Press Return to close this window."
    exit 1
}

# TraceISO requires Python 3.10 or newer (see pyproject.toml).
PYTHON_CMD=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            PYTHON_CMD="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.10 or newer was not found."
    echo
    echo "Install it from https://www.python.org/downloads/ (a normal macOS"
    echo "installer), then run this setup again."
    echo
    echo "The Python that ships with macOS is too old for TraceISO."
    echo
    read -r -p "Press Return to close this window."
    exit 1
fi

echo "[Setup] Using $("$PYTHON_CMD" --version 2>&1) at $(command -v "$PYTHON_CMD")"

if [ ! -x "venv/bin/python" ]; then
    echo "[Setup] Creating virtual environment..."
    "$PYTHON_CMD" -m venv venv || fail
fi

echo "[Setup] Updating pip..."
venv/bin/python -m pip install --upgrade pip || fail

echo "[Setup] Installing application dependencies..."
venv/bin/python -m pip install -r requirements.txt || fail

# The PyQt5 desktop tools are optional on macOS: the main Streamlit app does
# not need them, and no PyQt5 wheel exists for some Python/macOS combinations.
# A failure here is reported but does not fail the setup.
echo "[Setup] Installing optional GUI dependencies for the desktop tools..."
if venv/bin/python -m pip install --upgrade PyQt5 PyQt5-sip; then
    GUI_TOOLS_OK=1
elif venv/bin/python -m pip install "PyQt5==5.15.11" "PyQt5-sip"; then
    # The newest PyQt5 has no wheel for every Python/macOS combination, and pip
    # then tries to build Qt from source and fails. 5.15.11 is the last release
    # with broad macOS wheel coverage, including Apple Silicon.
    GUI_TOOLS_OK=1
    echo "[Setup] Installed PyQt5 5.15.11 (the newest release has no wheel here)."
else
    GUI_TOOLS_OK=0
    echo
    echo "[Setup] Note: PyQt5 could not be installed on this Mac."
    echo "[Setup] This usually means your Python is newer than the available"
    echo "[Setup] PyQt5 wheels. Python 3.12 is the safest version for the tools."
    echo "[Setup] TraceISO itself will still run. Only the separate desktop"
    echo "[Setup] tools (CRM Library Manager, Neptune Data Extractor, Neptune"
    echo "[Setup] Blank Restorer, Global Uncertainty Manager) are unavailable."
    echo
fi

echo "[Setup] Checking dependency compatibility..."
venv/bin/python -m pip check || fail

cp -f requirements.txt venv/.requirements.snapshot

# Keep the launchers double-clickable even if the executable bit was lost
# (for example when the project was delivered as a .zip).
chmod +x ./*.command >/dev/null 2>&1

echo
echo "Setup completed successfully."
if [ "$GUI_TOOLS_OK" -eq 1 ]; then
    echo "The desktop tools are available as well."
fi
echo "Start the app with 01_TraceISO_Launcher.command."
echo
read -r -p "Press Return to close this window."
exit 0
