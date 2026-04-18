import json
from datetime import timedelta

# path to your JSON file
JSON_PATH = "match13_rallies_behavior.json"


def sec_to_hms(seconds):
    """Convert seconds -> HH:MM:SS"""
    td = timedelta(seconds=int(seconds))
    total = int(td.total_seconds())

    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60

    return f"{h:02}:{m:02}:{s:02}"


with open(JSON_PATH, "r") as f:
    rallies = json.load(f)

for i, r in enumerate(rallies, 1):
    start = sec_to_hms(r["start"])
    end = sec_to_hms(r["end"])
    duration = sec_to_hms(r["duration"])

    print(f"RALLY {i}: {start}-{end}  |  duration: {duration}")