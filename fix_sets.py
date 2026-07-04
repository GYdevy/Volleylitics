import json
import re
from pathlib import Path
from typing import Dict, Any


RALLY_OUTPUT_ROOT = Path("/home/goshay/projects/Volleylitics/rally_segmentator/output")
HEATMAP_ROOT = Path("/home/goshay/projects/Volleylitics/heatmaps")


def build_clip_to_set_map(rallies_with_clips: list[dict[str, Any]]) -> Dict[str, int | None]:
    """
    Build a mapping like:
      'rally_000.mp4' -> 1
    from rallies_with_clips.json entries.
    """
    mapping: Dict[str, int | None] = {}

    for entry in rallies_with_clips:
        clip_path = entry.get("clip_path")
        set_id = entry.get("set_id", entry.get("set"))

        if not clip_path:
            continue

        clip_name = Path(clip_path).name
        mapping[clip_name] = set_id

    return mapping


def fix_match(match_id: str) -> None:
    rallies_path = RALLY_OUTPUT_ROOT / match_id / "rallies_with_clips.json"
    heatmap_path = HEATMAP_ROOT / match_id / "rally_results.json"

    if not rallies_path.exists():
        print(f"Skipping {match_id}: missing {rallies_path}")
        return

    if not heatmap_path.exists():
        print(f"Skipping {match_id}: missing {heatmap_path}")
        return

    with rallies_path.open("r", encoding="utf-8") as f:
        rallies_data = json.load(f)

    with heatmap_path.open("r", encoding="utf-8") as f:
        heatmap_data = json.load(f)

    if not isinstance(rallies_data, list):
        print(f"Skipping {match_id}: rallies_with_clips.json is not a list")
        return

    if not isinstance(heatmap_data, list):
        print(f"Skipping {match_id}: rally_results.json is not a list")
        return

    clip_to_set = build_clip_to_set_map(rallies_data)

    updated = 0
    missing = 0

    for entry in heatmap_data:
        clip_name = entry.get("clip_name")
        if not clip_name:
            missing += 1
            continue

        set_id = clip_to_set.get(clip_name)

        if set_id is None and clip_name not in clip_to_set:
            missing += 1
            continue

        entry["set_id"] = set_id
        entry["set"] = set_id
        updated += 1

    with heatmap_path.open("w", encoding="utf-8") as f:
        json.dump(heatmap_data, f, indent=2, ensure_ascii=False)

    print(f"{match_id}: updated {updated} entries, {missing} unmatched")


def main() -> None:
    match_pattern = re.compile(r"^match\d+$")

    match_ids = sorted(
        p.name for p in RALLY_OUTPUT_ROOT.iterdir()
        if p.is_dir() and match_pattern.match(p.name)
    )

    if not match_ids:
        print("No matchXX directories found.")
        return

    for match_id in match_ids:
        fix_match(match_id)


if __name__ == "__main__":
    main()
