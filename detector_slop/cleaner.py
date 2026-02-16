import json

JSON_PATH = "whistles_match11.json"
THRESHOLD = 2.0  # seconds

with open(JSON_PATH, "r") as f:
    whistles = json.load(f)

# Sort by time
whistles = sorted(whistles, key=lambda x: x["time"])

clusters = []
current_cluster = [whistles[0]]

for i in range(1, len(whistles)):
    prev = whistles[i - 1]
    curr = whistles[i]

    if curr["time"] - prev["time"] <= THRESHOLD:
        current_cluster.append(curr)
    else:
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        current_cluster = [curr]

# Catch last cluster
if len(current_cluster) > 1:
    clusters.append(current_cluster)

# Print results
if not clusters:
    print("No suspicious whistle clusters found.")
else:
    print(f"Found {len(clusters)} suspicious clusters:\n")

    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx + 1}:")
        for w in cluster:
            print(f"  ID {w['whistle_id']} @ {w['time']}s")
        print()
