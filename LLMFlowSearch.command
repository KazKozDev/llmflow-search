#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "========================================"
echo "   🚀 LLMFlow Search Agent Launcher"
echo "========================================"

# 1. Kill any process running on port 8000
echo "🧹 Cleaning up port 8000..."
PID=$(lsof -ti:8000)
if [ ! -z "$PID" ]; then
    kill -9 $PID
    echo "   - Killed process $PID on port 8000"
else
    echo "   - Port 8000 is free"
fi

# 2. Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "🔌 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system python3."
fi

# 3. Open browser in background after 3 seconds
(sleep 3 && open "http://localhost:8000") &

# 4. Start the server
echo "⚡ Starting Web Server..."
echo "   - Logs will appear below"
echo "   - Press Ctrl+C to stop"
echo "========================================"
echo ""

python3 web_server.py
