import json
from collections import defaultdict

GT_FILE = "rally_segmentator/output/match18/match18_rallies_gt.json"

with open(GT_FILE, "r") as f:
    rallies = json.load(f)

start_counts = defaultdict(list)
end_counts = defaultdict(list)

# collect timestamps
for i, r in enumerate(rallies):
    start = round(r["start"], 3)
    end = round(r["end"], 3)

    start_counts[start].append(i)
    end_counts[end].append(i)

print("=== Duplicate START times ===")
for t, idxs in start_counts.items():
    if len(idxs) > 1:
        print(f"time={t} → rallies={idxs}")

print("\n=== Duplicate END times ===")
for t, idxs in end_counts.items():
    if len(idxs) > 1:
        print(f"time={t} → rallies={idxs}")

# optional: start == end collisions (more interesting)
print("\n=== START == END collisions ===")
end_set = set(end_counts.keys())

for t, idxs in start_counts.items():
    if t in end_set:
        print(f"time={t} appears as START and END")
