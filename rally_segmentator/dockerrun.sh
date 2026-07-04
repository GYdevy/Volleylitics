#!/bin/bash
set -e

MATCH_ID="${1:?Usage: $0 MATCH_ID}"

echo "Running pipeline in Docker for $MATCH_ID..."

to_seconds() {
    local t="$1"
    IFS=: read -r h m s <<< "$t"
    echo $((10#$h * 3600 + 10#$m * 60 + 10#$s))
}

read -p "How many sets? " SET_COUNT

SEGMENTS_JSON="["
for ((i=1; i<=SET_COUNT; i++)); do
    read -p "Set ${i} start (HH:MM:SS): " START_STR
    read -p "Set ${i} end   (HH:MM:SS): " END_STR

    START_SEC=$(to_seconds "$START_STR")
    END_SEC=$(to_seconds "$END_STR")

    if [ "$i" -gt 1 ]; then
        SEGMENTS_JSON+=","
    fi

    SEGMENTS_JSON+="[$START_SEC,$END_SEC]"
done
SEGMENTS_JSON+="]"

echo "Using segments: $SEGMENTS_JSON"

docker run -it \
    -u "$(id -u):$(id -g)" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size=8G \
    -e SEGMENTS_JSON="$SEGMENTS_JSON" \
    -v ~/projects/Volleylitics:/workspace \
    -v /mnt/hdd/videos:/videos \
    -v /mnt/hdd/datasets:/datasets \
    volleylitics \
    python -m rally_segmentator.rally_timeline_maker --match-id "$MATCH_ID"

echo "Docker pipeline finished"

read -p "Review HITLs now? [y/n]: " answer

if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "Launching HITL reviewer locally..."
    $(pwd)/.venv/bin/python -m rally_segmentator.output.hitl_reviewer --match-id "$MATCH_ID"
else
    echo "Skipping HITL review"
fi
