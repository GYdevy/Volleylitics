#!/usr/bin/env bash
set -e

MATCH_ID="${1:?Usage: $0 MATCH_ID}"
VIDEO_PATH="/mnt/hdd/videos/${MATCH_ID}.mp4"

echo "=== Stage 1: Rally timeline ==="
bash rally_segmentator/dockerrun.sh "$MATCH_ID"

echo "=== Stage 2: Split rallies into clips ==="
python -m rally_segmentator.split_rallies --match-id "$MATCH_ID" --video "$VIDEO_PATH"

echo "=== Stage 3: Heatmaps ==="
bash heatmaps/run_heatmaps.bash "$MATCH_ID"

echo "=== Done ==="
