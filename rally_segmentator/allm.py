import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_event_time(event: Dict[str, Any]) -> Optional[float]:
    """
    Prefer anchored time, then raw time, then generic time field.
    """
    for key in ("t_anchor", "t_raw", "time"):
        value = event.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def normalize_type(event: Dict[str, Any]) -> Optional[str]:
    value = event.get("type")
    if not value:
        return None
    return str(value).strip().lower()


def build_rallies_for_match(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build rallies by stitching:
      serve -> next rally_end
    Ignore:
      other
    Skip malformed/incomplete sequences.
    """
    cleaned: List[Dict[str, Any]] = []

    for event in events:
        event_type = normalize_type(event)
        event_time = get_event_time(event)

        if event_type not in {"serve", "rally_end", "other"}:
            continue
        if event_time is None:
            continue

        cleaned.append({
            **event,
            "_type": event_type,
            "_time": event_time,
        })

    cleaned.sort(key=lambda e: e["_time"])

    rallies: List[Dict[str, Any]] = []
    current_serve: Optional[Dict[str, Any]] = None

    for event in cleaned:
        event_type = event["_type"]
        event_time = event["_time"]

        if event_type == "other":
            continue

        if event_type == "serve":
            # If we see a new serve before closing the previous one,
            # replace the previous open serve.
            current_serve = event
            continue

        if event_type == "rally_end":
            if current_serve is None:
                continue

            start = current_serve["_time"]
            end = event_time

            if end <= start:
                current_serve = None
                continue

            rally = {
                "start": start,
                "end": end,
                "duration": end - start,
                "label": "GT",
                "set_id": current_serve.get("set_id", current_serve.get("set")),
                "source": "whistles_deduped",
                "serve_whistle_id": current_serve.get("whistle_id"),
                "rally_end_whistle_id": event.get("whistle_id"),
            }

            rallies.append(rally)
            current_serve = None

    return rallies


def group_by_match(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for event in events:
        match_id = event.get("match_id")
        if match_id is None:
            continue
        match_id = str(match_id)
        grouped.setdefault(match_id, []).append(event)

    return grouped


def save_rallies(output_root: Path, match_id: str, rallies: List[Dict[str, Any]]) -> Path:
    match_dir = output_root / match_id
    match_dir.mkdir(parents=True, exist_ok=True)

    output_path = match_dir / "rallies_with_hitl.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rallies, f, indent=2, ensure_ascii=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create rallies_with_hitl.json per match from whistles_deduped.json"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to whistles_deduped.json",
    )
    parser.add_argument(
        "--output-root",
        default="rally_segmentator/output",
        help="Root output directory for per-match folders",
    )
    parser.add_argument(
        "--match-id",
        default=None,
        help="Optional single match_id to process",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_root)

    with input_path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    if not isinstance(events, list):
        raise ValueError("Input JSON must be a list of whistle events")

    grouped = group_by_match(events)

    if args.match_id is not None:
        grouped = {args.match_id: grouped.get(args.match_id, [])}

    if not grouped:
        print("No matches found.")
        return

    for match_id, match_events in grouped.items():
        rallies = build_rallies_for_match(match_events)
        output_path = save_rallies(output_root, match_id, rallies)
        print(f"{match_id}: wrote {len(rallies)} rallies -> {output_path}")


if __name__ == "__main__":
    main()
