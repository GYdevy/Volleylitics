import json
#from config import MATCH_ID
MATCH_ID = "match18"
INPUT_JSON = "rally_segmentator/anchored_matches/whistles_deduped.json"
OUTPUT_JSON = f"rally_segmentator/output/{MATCH_ID}/{MATCH_ID}_rallies_gt.json"


# =========================
# LOAD
# =========================
with open(INPUT_JSON, "r") as f:
    whistles = json.load(f)

# filter match
whistles = [w for w in whistles if w["match_id"] == MATCH_ID]

# sort by time
whistles.sort(key=lambda x: x["t_anchor"])

print("Total whistles:", len(whistles))
print("Types:", set(w["type"] for w in whistles))


# =========================
# BUILD RALLIES (serve → rally_end)
# =========================
rallies = []

i = 0
while i < len(whistles):

    w = whistles[i]
    w_type = w["type"].lower()

    # --------------------------
    # LOOK FOR SERVE
    # --------------------------
    if "serve" in w_type:

        start = w["t_anchor"]

        # search forward for next rally_end
        j = i + 1
        found = False

        while j < len(whistles):
            w_next = whistles[j]
            next_type = w_next["type"].lower()

            if "rally" in next_type:
                end = w_next["t_anchor"]

                if end > start:
                    rallies.append({
                        "start": start,
                        "end": end,
                        "duration": end - start
                    })

                found = True
                break

            j += 1

        # move pointer forward
        if found:
            i = j
        else:
            i += 1

    else:
        i += 1


# =========================
# SAVE
# =========================
with open(OUTPUT_JSON, "w") as f:
    json.dump(rallies, f, indent=2)

print("\nSaved rallies:", len(rallies))


# =========================
# STATS
# =========================
if rallies:
    durations = [r["duration"] for r in rallies]
    print("Avg duration:", sum(durations) / len(durations))
    print("Min duration:", min(durations))
    print("Max duration:", max(durations))
