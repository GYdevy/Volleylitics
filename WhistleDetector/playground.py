import csv
from collections import Counter, defaultdict
from pathlib import Path

CSV_PATH = Path(r"D:\Volleyballey\WhistleDetector\training_hitl.csv")

def find_unique_entries(csv_path):
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            rows.append(row)

    # assume UID is column 0, label is last column
    uid_counts = Counter(row[0] for row in rows)

    uniques = [row for row in rows if uid_counts[row[0]] == 1]

    print(f"Total rows        : {len(rows)}")
    print(f"Unique (appear 1x): {len(uniques)}\n")

    for row in uniques:
        uid = row[0]
        label = row[-1]
        print(f"{uid}  →  label={label}")

if __name__ == "__main__":
    find_unique_entries(CSV_PATH)
