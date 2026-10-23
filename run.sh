#!/usr/bin/env bash
# StrikeCast launcher for macOS / Linux.
#   ./run.sh          -> dashboard (http://localhost:8501)
#   ./run.sh api      -> REST API  (http://127.0.0.1:8000/docs)
#   ./run.sh test     -> run the test suite
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating local virtual environment (.venv) and installing packages..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

case "${1:-dashboard}" in
  api)  .venv/bin/python -m uvicorn app.main:app --port 8000 ;;
  test) .venv/bin/python -m pytest -q ;;
  *)    .venv/bin/python -m streamlit run dashboard/streamlit_app.py ;;
esac
