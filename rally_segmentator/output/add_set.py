import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

import json
from config import MATCH_ID, SEGMENTS

JSON_PATH = f"/workspace/rally_segmentator/output/{MATCH_ID}/rallies_with_clips.json"
OUT_PATH = JSON_PATH.replace(".json", "_with_sets.json")


# -------------------------
# helper
# -------------------------
def get_set_for_time(t, segments):
    for i, (start, end) in enumerate(segments):
        if start <= t <= end:
            return i + 1
    return None


# -------------------------
# load
# -------------------------
with open(JSON_PATH, "r") as f:
    data = json.load(f)


# -------------------------
# enrich
# -------------------------
for r in data:
    t = r["start"]  # already seconds

    r["set"] = get_set_for_time(t, SEGMENTS)


# -------------------------
# save
# -------------------------
with open(OUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Saved: {OUT_PATH}")





