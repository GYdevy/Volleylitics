#!/bin/bash

set -e  # stop on error

echo " Running pipeline in Docker..."

docker run \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size=8G \
    -v ~/projects/Volleylitics:/workspace \
    -v /mnt/hdd/videos:/videos \
    -v /mnt/hdd/datasets:/datasets \
    volleylitics \
    python -m rally_segmentator.rally_timeline_maker

echo " Docker pipeline finished"

read -p "Review HITLs now? [y/n]: " answer

if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo " Launching HITL reviewer locally..."
    python -m rally_segmentator.output.hitl_reviewer
else
    echo " Skipping HITL review"
fi
