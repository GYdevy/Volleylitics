import json
from config import MATCH_ID,OUTPUT_DIR
MATCH_DIR = OUTPUT_DIR / MATCH_ID

RALLIES_FILE = MATCH_DIR / "rallies.json"
HITL_FILE = MATCH_DIR / "hitl_decisions.json"
OUTPUT_FILE = MATCH_DIR / "rallies_with_hitl.json"


# =========================
# LOAD
# =========================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# =========================
# CHECK OVERLAP
# =========================
def overlaps(a, b):
    return not (a["end"] < b["start"] or b["end"] < a["start"])


# =========================
# MERGE INTERVALS
# =========================
def merge_intervals(r1, r2):
    return {
        "start": min(r1["start"], r2["start"]),
        "end": max(r1["end"], r2["end"]),
        "duration": max(r1["end"], r2["end"]) - min(r1["start"], r2["start"]),
        "label": "MERGED_HITL"
    }


# =========================
# MAIN MERGE LOGIC
# =========================
def add_hitl_to_rallies(rallies, hitl_decisions):

    # take only HITL marked as rally
    hitl_rallies = [
        {
            "start": h["start"],
            "end": h["end"],
            "duration": h["end"] - h["start"],
            "label": "HITL"
        }
        for h in hitl_decisions
        if h["decision"] == "rally"
    ]

    all_rallies = rallies.copy()

    for h in hitl_rallies:

        merged = False

        for i in range(len(all_rallies)):
            r = all_rallies[i]

            if overlaps(r, h):
                # merge into existing rally
                all_rallies[i] = merge_intervals(r, h)
                merged = True
                break

        if not merged:
            # add as new rally
            all_rallies.append(h)

    # =========================
    # FINAL CLEAN MERGE (IMPORTANT)
    # =========================

    all_rallies = sorted(all_rallies, key=lambda x: x["start"])

    merged_final = []

    for r in all_rallies:
        if not merged_final:
            merged_final.append(r)
            continue

        prev = merged_final[-1]

        if overlaps(prev, r) or r["start"] <= prev["end"] + 0.5:
            merged_final[-1] = merge_intervals(prev, r)
        else:
            merged_final.append(r)

    # recompute duration
    for r in merged_final:
        r["duration"] = r["end"] - r["start"]

    return merged_final


# =========================
# RUN
# =========================
if __name__ == "__main__":

    rallies = load_json(RALLIES_FILE)
    hitl = load_json(HITL_FILE)

    print("Original rallies:", len(rallies))

    merged = add_hitl_to_rallies(rallies, hitl)

    print("After HITL merge:", len(merged))

    with open(OUTPUT_FILE, "w") as f:
        json.dump(merged, f, indent=2)

    print("✅ Saved to", OUTPUT_FILE)
