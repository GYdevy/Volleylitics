import json
import os

# ===============================
# CONFIG
# ===============================

INPUT_FILES = [
    "whistles_match4.json",
    "whistles_match11.json",
    "whistles_match3.json"  # add/remove as needed
]

OUTPUT_FILE = "whistles_all.json"

REASSIGN_GLOBAL_IDS = True   # recommended

# ===============================
# LOAD + MERGE
# ===============================

all_whistles = []

for file in INPUT_FILES:
    if not os.path.exists(file):
        print(f"⚠ Skipping missing file: {file}")
        continue

    with open(file, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} whistles from {file}")

    all_whistles.extend(data)

print(f"\nTotal whistles before processing: {len(all_whistles)}")



# Ensure raw timestamp field exists for downstream anchoring
for w in all_whistles:
    if "t_raw" not in w:
        w["t_raw"] = w.get("time")

# ===============================
# SORT
# ===============================

all_whistles.sort(key=lambda w: (w["match_id"], w.get("t_raw", w.get("time", 0))))

# ===============================
# OPTIONAL: REASSIGN GLOBAL IDS
# ===============================

if REASSIGN_GLOBAL_IDS:
    for idx, w in enumerate(all_whistles):
        w["global_id"] = idx

# ===============================
# SAVE
# ===============================

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_whistles, f, indent=4)

print(f"\nSaved merged dataset → {OUTPUT_FILE}")
print(f"Final whistle count: {len(all_whistles)}")
