#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -x .venv-linux/bin/python ]]; then
    echo "Linux environment not found; running automatic setup..."
    "$SCRIPT_DIR/setup_linux.sh" --backend auto
fi

exec "$SCRIPT_DIR/.venv-linux/bin/python" "$SCRIPT_DIR/run_pipeline.py" "$@"
