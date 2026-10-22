#!/usr/bin/env zsh
set -e

echo "Starting ThesisOut Top-Science Platform..."
cd "$(dirname "$0")/backend"
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

cd "$(dirname "$0")/frontend"
npm run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

echo "ThesisOut Backend API: http://127.0.0.1:8000/docs"
echo "ThesisOut Web Dashboard: http://127.0.0.1:5173"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
