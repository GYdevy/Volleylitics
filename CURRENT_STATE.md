# Current State

## Working
- ROCm works
- VolleyVision player detector works
- Player detection sees the needed players
- ByteTrack works but creates fragmented IDs
- action_recog.pt exists but needs testing
- ball detection model exists but can improve

## Current problem
ByteTrack IDs are temporary and fragment over time.
Need identity layer above tracking.

## Do not solve yet
- full multicam
- full architecture
- robust Re-ID
- jersey digit model
- reports

## Next tiny task
Test action_recog.pt on one clip and save boxes around detected actions.
