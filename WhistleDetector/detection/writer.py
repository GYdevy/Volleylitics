import csv
from pathlib import Path

def append_timeline_row(
    csv_path: Path,
    uid: str,
    start: float,
    end: float,
    label: str,
    confidence: float,
    source: str,
):
    header = [
        "uid",
        "start_sec",
        "end_sec",
        "label",
        "confidence",
        "source",
    ]

    exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)

        writer.writerow([
            uid,
            f"{start:.3f}",
            f"{end:.3f}",
            label,
            f"{confidence:.4f}",
            source,
        ])
