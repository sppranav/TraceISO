#!/bin/bash
# CRM Library Manager - Launcher (macOS)
#
# Double-click this file in Finder, or run it from Terminal.
# It is the macOS counterpart of 02_CRM_Library_Manager_Launcher.bat.

cd "$(dirname "$0")" || exit 1

setup_required() {
    echo "CRM Library Manager is not set up yet, or its GUI dependency is missing."
    echo "Run 00_Setup_Update_Dependencies.command, then launch it again."
    echo
    echo "If setup reported that PyQt5 could not be installed, the desktop"
    echo "tools are unavailable on this Mac. TraceISO itself still works."
    echo
    read -r -p "Press Return to close this window."
    exit 1
}

[ -x "venv/bin/python" ] || setup_required

venv/bin/python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyQt5') else 1)" >/dev/null 2>&1 || setup_required

# Detach the tool from this Terminal window, mirroring the Windows launcher's
# use of pythonw. Output goes to a log file so a startup failure is not lost.
LOG="${TMPDIR:-/tmp}/traceiso_tools_crm_manager.log"
nohup venv/bin/python -m tools.crm_manager >"$LOG" 2>&1 &
TOOL_PID=$!

echo "Starting CRM Library Manager..."

# Surface an immediate startup failure instead of leaving an empty screen.
sleep 2
if ! kill -0 "$TOOL_PID" 2>/dev/null; then
    wait "$TOOL_PID"
    echo
    echo "CRM Library Manager did not start. Details:"
    echo
    cat "$LOG"
    echo
    read -r -p "Press Return to close this window."
    exit 1
fi

echo "CRM Library Manager is running. You can close this window."
exit 0
