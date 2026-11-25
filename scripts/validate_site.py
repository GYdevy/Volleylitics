#!/usr/bin/env python3
"""Validate the static Volleylitics match catalog before deployment."""

from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = PROJECT_ROOT / "docs"
INDEX_PATH = DOCS_ROOT / "data" / "matches" / "index.json"
REQUIRED_MATCH_FIELDS = {
    "id",
    "label",
    "videoId",
    "longestRally",
    "ralliesFile",
    "detailsFile",
    "date",
    "venue",
    "competition",
    "result",
    "setScore",
}
# These two old files use their first "team" row as a set-number header rather
# than score data. Keep the exception narrow so newly published totals remain
# strictly validated.
LEGACY_HEADER_ROWS = {("match11", 0), ("match15", 0)}


class ValidationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"Missing file: {path.relative_to(PROJECT_ROOT)}") from exc
    except json.JSONDecodeError as exc:
        relative = path.relative_to(PROJECT_ROOT)
        raise ValidationError(f"Invalid JSON in {relative}:{exc.lineno}: {exc.msg}") from exc


def resolve_site_path(value: str, field: str, match_id: str) -> Path:
    require(isinstance(value, str) and value.startswith("./"), f"{match_id}: {field} must start with ./")
    relative = Path(value[2:])
    require(not relative.is_absolute() and ".." not in relative.parts, f"{match_id}: unsafe {field} path")
    path = (DOCS_ROOT / relative).resolve()
    require(DOCS_ROOT.resolve() in path.parents, f"{match_id}: {field} escapes docs/")
    return path


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_point(value: Any, label: str) -> None:
    if value is None:
        return
    require(isinstance(value, list) and len(value) == 2, f"{label} must be null or [x, y]")
    require(all(finite_number(item) for item in value), f"{label} contains a non-finite coordinate")


def validate_rallies(match_id: str, path: Path) -> int:
    rallies = load_json(path)
    require(isinstance(rallies, list) and rallies, f"{match_id}: rally file must be a non-empty list")
    seen_ids: set[int] = set()
    for position, rally in enumerate(rallies):
        label = f"{match_id}: rally[{position}]"
        require(isinstance(rally, dict), f"{label} must be an object")
        rally_id = rally.get("rally_id")
        require(isinstance(rally_id, int) and not isinstance(rally_id, bool), f"{label} has invalid rally_id")
        require(rally_id not in seen_ids, f"{match_id}: duplicate rally_id {rally_id}")
        seen_ids.add(rally_id)
        start = rally.get("start")
        end = rally.get("end")
        require(finite_number(start) and start >= 0, f"{label} has invalid start")
        require(finite_number(end) and end > start, f"{label} must end after it starts")
        set_id = rally.get("set_id", rally.get("set"))
        require(set_id is None or (isinstance(set_id, int) and set_id > 0), f"{label} has invalid set_id")
        require(isinstance(rally.get("clip_name"), str) and rally["clip_name"], f"{label} has no clip_name")
        validate_point(rally.get("attack_point"), f"{label}.attack_point")
        validate_point(rally.get("landing_point"), f"{label}.landing_point")
    return len(rallies)


def validate_details(match_id: str, path: Path, warnings: list[str]) -> None:
    details = load_json(path)
    require(isinstance(details, dict), f"{match_id}: details must be an object")
    teams = details.get("teams")
    require(isinstance(teams, list) and len(teams) == 2, f"{match_id}: details must contain two teams")
    for position, team in enumerate(teams):
        label = f"{match_id}: team[{position}]"
        require(isinstance(team, dict), f"{label} must be an object")
        require(isinstance(team.get("name"), str) and team["name"].strip(), f"{label} has no name")
        sets = team.get("sets")
        require(isinstance(sets, list) and sets, f"{label} has no set scores")
        require(all(isinstance(score, int) and score >= 0 for score in sets), f"{label} has invalid set scores")
        if (match_id, position) in LEGACY_HEADER_ROWS:
            if team.get("total") != sum(sets):
                warnings.append(f"{label} is a legacy set-number header, not score data")
        else:
            require(team.get("total") == sum(sets), f"{label} total does not equal its set scores")


def validate_site() -> tuple[int, int, list[str]]:
    require((DOCS_ROOT / "index.html").is_file(), "docs/index.html is missing")
    catalog = load_json(INDEX_PATH)
    require(isinstance(catalog, list) and catalog, "Match index must be a non-empty list")

    warnings: list[str] = []
    seen_matches: set[str] = set()
    match_count = 0
    rally_count = 0
    for group_position, group in enumerate(catalog):
        require(isinstance(group, dict), f"season[{group_position}] must be an object")
        require(isinstance(group.get("season"), str) and group["season"].strip(), f"season[{group_position}] has no name")
        matches = group.get("matches")
        require(isinstance(matches, list) and matches, f"{group['season']}: matches must be a non-empty list")
        for match in matches:
            require(isinstance(match, dict), f"{group['season']}: match entry must be an object")
            missing = REQUIRED_MATCH_FIELDS - match.keys()
            require(not missing, f"Match entry is missing: {', '.join(sorted(missing))}")
            match_id = match["id"]
            require(isinstance(match_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", match_id), f"Invalid match id: {match_id!r}")
            require(match_id not in seen_matches, f"Duplicate match id: {match_id}")
            seen_matches.add(match_id)
            match_count += 1

            require(isinstance(match["label"], str) and match["label"].strip(), f"{match_id}: label is empty")
            require(re.fullmatch(r"[A-Za-z0-9_-]{6,20}", match["videoId"] or ""), f"{match_id}: invalid YouTube videoId")
            require(re.fullmatch(r"\d{2}:\d{2}", match["longestRally"] or ""), f"{match_id}: invalid longestRally")
            require(re.fullmatch(r"\d+-\d+", match["setScore"] or ""), f"{match_id}: invalid setScore")
            try:
                date.fromisoformat(match["date"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{match_id}: date must use YYYY-MM-DD") from exc

            rallies_path = resolve_site_path(match["ralliesFile"], "ralliesFile", match_id)
            details_path = resolve_site_path(match["detailsFile"], "detailsFile", match_id)
            expected_dir = (DOCS_ROOT / "data" / "matches" / match_id).resolve()
            require(rallies_path.parent == expected_dir, f"{match_id}: ralliesFile points to another match")
            require(details_path.parent == expected_dir, f"{match_id}: detailsFile points to another match")
            rally_count += validate_rallies(match_id, rallies_path)
            validate_details(match_id, details_path, warnings)

            photo = match.get("photo")
            if isinstance(photo, str) and photo.startswith("./"):
                photo_path = resolve_site_path(photo, "photo", match_id)
                if not photo_path.is_file():
                    warnings.append(f"{match_id}: optional photo is missing: {photo}")

    return match_count, rally_count, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        match_count, rally_count, warnings = validate_site()
    except ValidationError as exc:
        print(f"SITE VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Site data valid: {match_count} matches, {rally_count} rallies, {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
