import json
import re
from pathlib import Path


OUTPUT_ROOT = Path("/home/goshay/projects/Volleylitics/rally_segmentator/output")
SKIP_MATCHES = {"match14", "match17", "match18"}


def hhmmss_to_seconds(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {value!r}. Expected HH:MM:SS")

    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + int(s)


def find_match_dirs(root: Path):
    pattern = re.compile(r"^match\d+$")
    matches = []

    for path in sorted(root.iterdir()):
        if path.is_dir() and pattern.match(path.name) and path.name not in SKIP_MATCHES:
            matches.append(path)

    return matches


def ask_set_segments(match_name: str):
    while True:
        raw = input(f"\n{match_name} - how many sets? ").strip()
        try:
            num_sets = int(raw)
            if num_sets <= 0:
                raise ValueError
            break
        except ValueError:
            print("Please enter a positive integer.")

    segments = []

    for set_id in range(1, num_sets + 1):
        while True:
            try:
                start_str = input(f"Set {set_id} start (HH:MM:SS): ").strip()
                end_str = input(f"Set {set_id} end   (HH:MM:SS): ").strip()

                start_sec = hhmmss_to_seconds(start_str)
                end_sec = hhmmss_to_seconds(end_str)

                if end_sec <= start_sec:
                    print("End must be after start. Try again.")
                    continue

                segments.append({
                    "set_id": set_id,
                    "start": start_sec,
                    "end": end_sec,
                })
                break
            except ValueError as e:
                print(e)

    return segments


def assign_set_id(rallies, segments):
    updated = 0
    unmatched = 0

    for rally in rallies:
        start = rally.get("start")
        if start is None:
            rally["set_id"] = None
            unmatched += 1
            continue

        set_id = None
        for seg in segments:
            if seg["start"] <= start <= seg["end"]:
                set_id = seg["set_id"]
                break

        rally["set_id"] = set_id
        if set_id is None:
            unmatched += 1
        else:
            updated += 1

    return updated, unmatched


def main():
    match_dirs = find_match_dirs(OUTPUT_ROOT)

    if not match_dirs:
        print("No matching directories found.")
        return

    print("Matches to process:")
    for match_dir in match_dirs:
        print(f" - {match_dir.name}")

    for match_dir in match_dirs:
        json_path = match_dir / "rallies_with_hitl.json"

        if not json_path.exists():
            print(f"\nSkipping {match_dir.name}: no rallies_with_hitl.json")
            continue

        try:
            with json_path.open("r", encoding="utf-8") as f:
                rallies = json.load(f)
        except Exception as e:
            print(f"\nSkipping {match_dir.name}: failed to read JSON ({e})")
            continue

        if not isinstance(rallies, list):
            print(f"\nSkipping {match_dir.name}: rallies_with_hitl.json is not a list")
            continue

        print(f"\n=== {match_dir.name} ===")
        segments = ask_set_segments(match_dir.name)

        updated, unmatched = assign_set_id(rallies, segments)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rallies, f, indent=2, ensure_ascii=False)

        print(f"Saved {json_path}")
        print(f"Assigned set_id for {updated} rallies")
        if unmatched:
            print(f"{unmatched} rallies did not match any set segment")


if __name__ == "__main__":
    main()
