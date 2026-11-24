"""Prepare a processed Volleylitics match for the static site in ``docs``."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tkinter as tk
from datetime import date
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from urllib.parse import parse_qs, urlparse


DOCS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_ROOT.parent
MATCHES_ROOT = DOCS_ROOT / "data" / "matches"
INDEX_PATH = MATCHES_ROOT / "index.json"

COLORS = {
    "background": "#0b1020",
    "panel": "#121a2b",
    "input": "#172137",
    "text": "#f3f6fb",
    "muted": "#94a3b8",
    "green": "#22c55e",
    "blue": "#38bdf8",
}


def load_index() -> list[dict]:
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("docs/data/matches/index.json must contain a list")
    return data


def write_json(path: Path, data: object) -> None:
    """Write stable UTF-8/LF JSON on Windows and Unix."""
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(data, output, indent=2, ensure_ascii=False)
        output.write("\n")


def find_latest_output() -> Path:
    output_root = PROJECT_ROOT / "output"
    candidates = []
    if output_root.is_dir():
        for path in output_root.iterdir():
            results = path / "heatmap" / "rally_results.json"
            if path.is_dir() and results.is_file():
                candidates.append((results.stat().st_mtime, path))
    return max(candidates, default=(0, output_root))[1]


def parse_scores(value: str) -> list[int]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("Enter each team's set scores, separated by commas.")
    if len(pieces) > 5:
        raise ValueError("A match cannot contain more than five sets.")
    try:
        scores = [int(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError("Set scores must be whole numbers separated by commas.") from exc
    if any(score < 0 or score > 99 for score in scores):
        raise ValueError("Set scores must be between 0 and 99.")
    return scores


def extract_youtube_id(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Enter a YouTube video URL or video ID.")
    if re.fullmatch(r"[A-Za-z0-9_-]{6,20}", value):
        return value
    parsed = urlparse(value)
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/", "/live/")):
            candidate = parsed.path.strip("/").split("/")[1]
        else:
            candidate = ""
    else:
        candidate = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", candidate):
        raise ValueError("That does not look like a valid YouTube URL or video ID.")
    return candidate


def format_longest_rally(rallies: list[dict]) -> str:
    longest = max(
        (max(0.0, float(item.get("end", 0)) - float(item.get("start", 0))) for item in rallies),
        default=0.0,
    )
    total_seconds = int(round(longest))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def calculate_set_score(first: list[int], second: list[int]) -> str:
    first_wins = sum(a > b for a, b in zip(first, second))
    second_wins = sum(b > a for a, b in zip(first, second))
    high, low = sorted((first_wins, second_wins), reverse=True)
    return f"{high}-{low}"


def build_complete_rally_feed(source_dir: Path, tracked_rallies: list[dict]) -> list[dict]:
    """Merge heatmap results onto the complete clipped-rally timeline.

    The website can list rallies without coordinates; it only needs landing_point
    for heatmap dots.  Older analytics files omitted untracked rallies entirely,
    so publishing also repairs those older outputs without rerunning detection.
    """

    timeline_path = source_dir / "rallies_with_clips.json"
    if not timeline_path.is_file():
        return tracked_rallies
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not isinstance(timeline, list):
        raise ValueError("rallies_with_clips.json must contain a list")

    tracked_by_id = {
        int(item["rally_id"]): item
        for item in tracked_rallies
        if item.get("rally_id") is not None
    }
    complete = []
    for fallback_id, rally in enumerate(timeline):
        rally_id = int(rally.get("id", fallback_id))
        clip_path = rally.get("clip_path") or f"rally_{rally_id:03d}.mp4"
        merged = {
            "clip_name": Path(clip_path).name,
            "rally_id": rally_id,
            "start": rally.get("start"),
            "end": rally.get("end"),
            "set_id": rally.get("set"),
            "positions": [],
            "attack_point": None,
            "landing_point": None,
            "tracking_status": "not_tracked",
            "debug_output_path": None,
        }
        tracked = tracked_by_id.get(rally_id)
        if tracked:
            merged.update(tracked)
            merged.setdefault(
                "tracking_status",
                "tracked" if tracked.get("landing_point") else "not_tracked",
            )
        complete.append(merged)
    return complete


def publish_match(values: dict, overwrite: bool = False) -> Path:
    source_dir = Path(values["source_dir"]).expanduser().resolve()
    source_results = source_dir / "heatmap" / "rally_results.json"
    if not source_results.is_file():
        direct_results = source_dir / "rally_results.json"
        if direct_results.is_file():
            source_results = direct_results
        else:
            raise FileNotFoundError(
                "The selected output has no heatmap/rally_results.json file."
            )

    match_id = values["match_id"].strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", match_id):
        raise ValueError("Match ID may contain only letters, numbers, hyphens, and underscores.")
    label = values["label"].strip()
    season = values["season"].strip()
    team_one = values["team_one"].strip()
    team_two = values["team_two"].strip()
    if not all((label, season, team_one, team_two)):
        raise ValueError("Season, label, and both team names are required.")

    match_date = values["date"].strip()
    try:
        date.fromisoformat(match_date)
    except ValueError as exc:
        raise ValueError("Date must use YYYY-MM-DD format.") from exc

    first_scores = parse_scores(values["team_one_scores"])
    second_scores = parse_scores(values["team_two_scores"])
    if len(first_scores) != len(second_scores):
        raise ValueError("Both teams must have the same number of set scores.")

    video_id = extract_youtube_id(values["video"])
    tracked_rallies = json.loads(source_results.read_text(encoding="utf-8"))
    if not isinstance(tracked_rallies, list):
        raise ValueError("rally_results.json must contain a list.")
    rallies = build_complete_rally_feed(source_dir, tracked_rallies)
    if not isinstance(rallies, list) or not rallies:
        raise ValueError("rally_results.json must contain at least one rally.")

    index = load_index()
    existing = next(
        (
            match
            for group in index
            for match in group.get("matches", [])
            if match.get("id") == match_id
        ),
        None,
    )
    destination = MATCHES_ROOT / match_id
    if (destination.exists() or existing is not None) and not overwrite:
        raise FileExistsError(f"{match_id} already exists on the site.")

    details = {
        "teams": [
            {"name": team_one, "sets": first_scores, "total": sum(first_scores)},
            {"name": team_two, "sets": second_scores, "total": sum(second_scores)},
        ]
    }
    entry = {
        "id": match_id,
        "label": label,
        "videoId": video_id,
        "photo": values.get("photo", "").strip() or "./placeholder.svg",
        "longestRally": format_longest_rally(rallies),
        "ralliesFile": f"./data/matches/{match_id}/rally_results.json",
        "detailsFile": f"./data/matches/{match_id}/match_details.json",
        "date": match_date,
        "venue": values["venue"],
        "competition": values["competition"],
        "result": values["result"],
        "setScore": calculate_set_score(first_scores, second_scores),
    }

    for group in index:
        group["matches"] = [
            match for match in group.get("matches", []) if match.get("id") != match_id
        ]
    season_group = next((group for group in index if group.get("season") == season), None)
    if season_group is None:
        season_group = {"season": season, "matches": []}
        index.insert(0, season_group)
    season_group.setdefault("matches", []).insert(0, entry)
    index[:] = [group for group in index if group.get("matches")]

    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "rally_results.json", rallies)
    court_map = source_results.parent / "court_map_all.png"
    if court_map.is_file():
        shutil.copy2(court_map, destination / "court_map_all.png")
    write_json(destination / "match_details.json", details)
    write_json(INDEX_PATH, index)
    return destination


class AddMatchWindow(tk.Tk):
    def __init__(self, source_dir: Path | None = None) -> None:
        super().__init__()
        self.title("Add processed match to website")
        self.geometry("760x750")
        self.minsize(700, 690)
        self.configure(bg=COLORS["background"])

        index = load_index()
        default_season = str(index[0].get("season", "")) if index else ""
        source = source_dir.resolve() if source_dir else find_latest_output()
        self.vars = {
            "source_dir": tk.StringVar(value=str(source)),
            "match_id": tk.StringVar(value=source.name if source.is_dir() else "match"),
            "season": tk.StringVar(value=default_season),
            "label": tk.StringVar(),
            "video": tk.StringVar(),
            "date": tk.StringVar(value=date.today().isoformat()),
            "venue": tk.StringVar(value="home"),
            "competition": tk.StringVar(value="league"),
            "result": tk.StringVar(value="win"),
            "team_one": tk.StringVar(),
            "team_one_scores": tk.StringVar(),
            "team_two": tk.StringVar(),
            "team_two_scores": tk.StringVar(),
            "photo": tk.StringVar(value="./placeholder.svg"),
        }
        self._build()

    def _label(self, parent: tk.Misc, text: str, row: int, column: int = 0) -> None:
        tk.Label(
            parent, text=text, bg=COLORS["panel"], fg=COLORS["muted"],
            font=("Segoe UI", 8, "bold"),
        ).grid(row=row, column=column, sticky="w", padx=8, pady=(7, 2))

    def _entry(self, parent: tk.Misc, key: str, row: int, column: int = 0, span: int = 1):
        entry = tk.Entry(
            parent, textvariable=self.vars[key], bg=COLORS["input"],
            fg=COLORS["text"], insertbackground=COLORS["text"],
            relief="flat", font=("Segoe UI", 10),
        )
        entry.grid(row=row, column=column, columnspan=span, sticky="ew", padx=8, pady=(0, 5), ipady=7)
        return entry

    def _combo(self, parent: tk.Misc, key: str, values: tuple[str, ...], row: int, column: int):
        combo = ttk.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly")
        combo.grid(row=row, column=column, sticky="ew", padx=8, pady=(0, 5), ipady=4)

    def _build(self) -> None:
        tk.Label(
            self, text="ADD MATCH TO SITE", bg=COLORS["background"], fg=COLORS["green"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=22, pady=(16, 2))
        tk.Label(
            self, text="Prepare a processed match for the docs site",
            bg=COLORS["background"], fg=COLORS["text"], font=("Segoe UI", 19, "bold"),
        ).pack(anchor="w", padx=22, pady=(0, 10))

        panel = tk.Frame(self, bg=COLORS["panel"], padx=10, pady=8)
        panel.pack(fill="both", expand=True, padx=20, pady=(0, 18))
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)

        self._label(panel, "PROCESSED OUTPUT FOLDER", 0)
        source_entry = self._entry(panel, "source_dir", 1, span=2)
        source_entry.grid_configure(padx=(8, 105))
        browse = tk.Button(
            panel, text="Browse", command=self.browse, bg=COLORS["input"],
            fg=COLORS["text"], relief="flat", font=("Segoe UI", 9, "bold"),
        )
        browse.grid(row=1, column=1, sticky="e", padx=8, pady=(0, 5), ipadx=10, ipady=6)

        self._label(panel, "MATCH ID", 2, 0)
        self._label(panel, "SEASON", 2, 1)
        self._entry(panel, "match_id", 3, 0)
        self._entry(panel, "season", 3, 1)
        self._label(panel, "DISPLAY LABEL", 4)
        self._entry(panel, "label", 5, span=2)
        self._label(panel, "YOUTUBE URL OR VIDEO ID", 6)
        self._entry(panel, "video", 7, span=2)
        self._label(panel, "DATE (YYYY-MM-DD)", 8, 0)
        self._label(panel, "OPTIONAL PHOTO PATH", 8, 1)
        self._entry(panel, "date", 9, 0)
        self._entry(panel, "photo", 9, 1)
        self._label(panel, "VENUE", 10, 0)
        self._label(panel, "COMPETITION", 10, 1)
        self._combo(panel, "venue", ("home", "away", "neutral"), 11, 0)
        self._combo(panel, "competition", ("league", "cup", "friendly"), 11, 1)
        self._label(panel, "RESULT", 12, 0)
        self._combo(panel, "result", ("win", "loss"), 13, 0)
        self._label(panel, "TEAM 1 NAME", 14, 0)
        self._label(panel, "TEAM 1 SET SCORES (comma-separated)", 14, 1)
        self._entry(panel, "team_one", 15, 0)
        self._entry(panel, "team_one_scores", 15, 1)
        self._label(panel, "TEAM 2 NAME", 16, 0)
        self._label(panel, "TEAM 2 SET SCORES (comma-separated)", 16, 1)
        self._entry(panel, "team_two", 17, 0)
        self._entry(panel, "team_two_scores", 17, 1)

        tk.Label(
            panel,
            text="Copies rally analytics and the court map—not the large rally video clips.",
            bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 9),
        ).grid(row=18, column=0, columnspan=2, sticky="w", padx=8, pady=(10, 5))
        publish = tk.Button(
            panel, text="Add match to docs", command=self.publish,
            bg=COLORS["green"], fg="#07140b", activebackground=COLORS["blue"],
            relief="flat", font=("Segoe UI", 10, "bold"), cursor="hand2",
        )
        publish.grid(row=19, column=0, columnspan=2, sticky="ew", padx=8, pady=8, ipady=8)

    def browse(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose a processed match output folder",
            initialdir=self.vars["source_dir"].get() or str(PROJECT_ROOT / "output"),
        )
        if selected:
            path = Path(selected)
            self.vars["source_dir"].set(str(path))
            self.vars["match_id"].set(path.name)

    def publish(self) -> None:
        values = {key: variable.get() for key, variable in self.vars.items()}
        destination = MATCHES_ROOT / values["match_id"].strip()
        overwrite = False
        if destination.exists():
            overwrite = messagebox.askyesno(
                "Replace existing match?",
                f"{destination.name} already exists. Replace its site data and index entry?",
                parent=self,
            )
            if not overwrite:
                return
        try:
            result = publish_match(values, overwrite=overwrite)
        except Exception as exc:
            messagebox.showerror("Could not add match", str(exc), parent=self)
            return
        messagebox.showinfo(
            "Match added",
            f"Prepared {result.name} in docs/data/matches.\n\n"
            "Review the site changes, then commit and push them to deploy.",
            parent=self,
        )
        self.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Add processed match data to the docs site")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert extract_youtube_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert parse_scores("25, 21, 25") == [25, 21, 25]
        assert calculate_set_score([25, 21, 25], [20, 25, 22]) == "2-1"
        return 0
    try:
        app = AddMatchWindow(args.source_dir)
        app.mainloop()
        return 0
    except Exception as exc:
        try:
            messagebox.showerror("Add match error", str(exc))
        except Exception:
            print(f"Add match error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
