#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/fastapi_app"

exec ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --reload
