#!/usr/bin/env bash
set -e

MATCH_ID="${1:-match17}"
VIDEO_PATH="/run/media/gyank/HDD/videos/${MATCH_ID}.mp4"
CALIB_DIR="$HOME/projects/Volleylitics/heatmaps/calibration"
CALIB_PATH="${CALIB_DIR}/${MATCH_ID}.json"

mkdir -p "$CALIB_DIR"

if [ ! -f "$CALIB_PATH" ]; then
  echo "Calibration not found, launching point picker..."
  .venv/bin/python heatmaps/point.py \
    --video "$VIDEO_PATH" \
    --out "$CALIB_PATH" \
    --seek-minutes 10
else
  echo "Using existing calibration: $CALIB_PATH"
fi

docker run --rm \
  -w /workspace \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v "$HOME/projects/Volleylitics/heatmaps/calibration:/workspace/heatmaps/calibration" \
  -v "$HOME/projects/Volleylitics:/workspace" \
  volleylitics \
  python -u -m heatmaps.generate_heatmap \
    --match-id "$MATCH_ID" \
    --calibration "/workspace/heatmaps/calibration/${MATCH_ID}.json"
