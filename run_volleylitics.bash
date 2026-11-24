#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "run_volleylitics.bash is retained as a compatibility wrapper." >&2
echo "Use run_linux.sh for new Linux runs." >&2
exec "$SCRIPT_DIR/run_linux.sh" "$@"
