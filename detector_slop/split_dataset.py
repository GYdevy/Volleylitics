import json
from collections import Counter

INPUT_JSON = "whistles_all.json"

TRAIN_MATCH = "match4"
VAL_MATCH   = "match11"
TEST_MATCH  = "match3"

OUT_TRAIN = "whistles_train.json"
OUT_VAL   = "whistles_val.json"
OUT_TEST  = "whistles_test.json"


# ===============================
# LOAD DATA
# ===============================
with open(INPUT_JSON, "r") as f:
    whistles = json.load(f)

print(f"Total whistles: {len(whistles)}")

# ===============================
# SPLIT
# ===============================
train = [w for w in whistles if w["match_id"] == TRAIN_MATCH]
val   = [w for w in whistles if w["match_id"] == VAL_MATCH]
test  = [w for w in whistles if w["match_id"] == TEST_MATCH]

# ===============================
# SAVE
# ===============================
with open(OUT_TRAIN, "w") as f:
    json.dump(train, f, indent=4)

with open(OUT_VAL, "w") as f:
    json.dump(val, f, indent=4)

with open(OUT_TEST, "w") as f:
    json.dump(test, f, indent=4)

print("\nSaved splits:")
print(f"Train: {len(train)}")
print(f"Val:   {len(val)}")
print(f"Test:  {len(test)}")


# ===============================
# CLASS DISTRIBUTION
# ===============================
def print_stats(name, data):
    types = [w["type"] for w in data]
    c = Counter(types)
    print(f"\n{name} distribution:")
    for k, v in c.items():
        print(f"  {k}: {v}")

print_stats("TRAIN", train)
print_stats("VAL", val)
print_stats("TEST", test)
