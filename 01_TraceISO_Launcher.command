#!/bin/bash
# TraceISO - Launcher (macOS)
#
# Double-click this file in Finder, or run it from Terminal.
# It is the macOS counterpart of 01_TraceISO_Launcher.bat.
#
# On Windows the launcher opens a small PyQt setup window. On macOS it starts
# the same Streamlit app directly in this Terminal window and opens your
# browser. Keep this window open while you work; press Control-C to stop.

cd "$(dirname "$0")" || exit 1

if [ ! -x "venv/bin/python" ]; then
    echo "TraceISO is not set up yet."
    echo "Run 00_Setup_Update_Dependencies.command, then launch TraceISO again."
    echo
    read -r -p "Press Return to close this window."
    exit 1
fi

if ! venv/bin/python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('streamlit') else 1)" >/dev/null 2>&1; then
    echo "TraceISO's dependencies are missing or incomplete."
    echo "Run 00_Setup_Update_Dependencies.command, then launch TraceISO again."
    echo
    read -r -p "Press Return to close this window."
    exit 1
fi

# An occupied port may belong to another application. Start our own process
# on a free loopback port instead of assuming an existing listener is TraceISO.
PORT="$(venv/bin/python -c "
import socket
for port in (8501, 8502, 8503, 0):
    s = socket.socket()
    try:
        s.bind(('127.0.0.1', port))
    except OSError:
        s.close()
        continue
    print(s.getsockname()[1])
    s.close()
    break
")"

if [ -z "$PORT" ]; then
    echo "Error: no local port was available for TraceISO."
    echo "Close other TraceISO windows and try again."
    echo
    read -r -p "Press Return to close this window."
    exit 1
fi

echo "==================================================="
echo "     TraceISO"
echo "==================================================="
echo
echo "Starting on http://localhost:$PORT"
echo "Keep this window open while you work. Press Control-C to stop TraceISO."
echo

venv/bin/python -m streamlit run TraceISO.py --server.address 127.0.0.1 --server.port "$PORT" &
STREAMLIT_PID=$!

# Open the browser after the selected server answers Streamlit's health check.
for _ in $(seq 1 40); do
    if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        break
    fi
    if venv/bin/python -c "
import sys
from urllib.request import urlopen
try:
    with urlopen('http://127.0.0.1:$PORT/_stcore/health', timeout=0.5) as response:
        ready = response.status == 200 and response.read().strip() == b'ok'
except Exception:
    ready = False
sys.exit(0 if ready else 1)
" >/dev/null 2>&1; then
        open "http://localhost:$PORT"
        break
    fi
    sleep 0.5
done

wait "$STREAMLIT_PID"
STATUS=$?

if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 130 ]; then
    echo
    echo "TraceISO stopped unexpectedly. The messages above describe the error."
    echo
    read -r -p "Press Return to close this window."
fi

exit "$STATUS"
