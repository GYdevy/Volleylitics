#!/usr/bin/env bash
set -e

MATCH_ID="${1:?Usage: $0 MATCH_ID}"

PROJECT_DIR="$HOME/projects/Volleylitics"
VIDEO_PATH="$PROJECT_DIR/videos/${MATCH_ID}.mp4"
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [ ! -f "$VIDEO_PATH" ]; then
  echo "Video not found: $VIDEO_PATH"
  exit 1
fi

echo "=== Stage 1: Rally timeline ==="
bash rally_segmentator/dockerrun.sh "$MATCH_ID"

echo "=== Stage 2: Split rallies into clips ==="
"$PYTHON" -m rally_segmentator.split_rallies \
  --match-id "$MATCH_ID" \
  --video "$VIDEO_PATH"

echo "=== Stage 3: Heatmaps ==="
bash heatmaps/run_heatmaps.bash "$MATCH_ID"

echo "=== Done ==="
