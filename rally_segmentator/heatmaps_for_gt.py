import shutil
import subprocess
from pathlib import Path

ROOT = Path("/home/goshay/projects/Volleylitics")
OUTPUT_ROOT = ROOT / "rally_segmentator" / "output"
CALIB_ROOT = ROOT / "heatmaps" / "calibration"
VIDEO_ROOT = Path("/mnt/hdd/videos")

SKIP_MATCHES = {"match14", "match17", "match18"}  # example: {"match14", "match17"}

def run(cmd, cwd=None):
    print("\n>>>", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    match_dirs = sorted(
        p for p in OUTPUT_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("match") and p.name not in SKIP_MATCHES
    )

    for match_dir in match_dirs:
        match_id = match_dir.name
        rallies_file = match_dir / "rallies_with_hitl.json"
        clips_dir = match_dir / "rally_clips"
        calib_file = CALIB_ROOT / f"{match_id}.json"
        video_file = VIDEO_ROOT / f"{match_id}.mp4"
        final_results = ROOT / "heatmaps" / match_id / "rally_results.json"

        print(f"\n====================")
        print(f"Processing {match_id}")
        print(f"====================")

        if not rallies_file.exists():
            print(f"Skipping {match_id}: no rallies_with_hitl.json")
            continue

        if not video_file.exists():
            print(f"Skipping {match_id}: video not found at {video_file}")
            continue

        if final_results.exists():
            ans = input(f"{match_id} already has rally_results.json. Rebuild? [y/N]: ").strip().lower()
            if ans != "y":
                print(f"Skipping {match_id}")
                continue

        if not calib_file.exists():
            print(f"Calibration missing for {match_id}. Launching point picker...")
            run([
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "heatmaps" / "point.py"),
                "--video", str(video_file),
                "--out", str(calib_file),
                "--seek-minutes", "10",
            ], cwd=ROOT)

            if not calib_file.exists():
                print(f"Skipping {match_id}: calibration still missing after point picker")
                continue

        # create clips
        run([
            "python",
            "-m",
            "rally_segmentator.split_rallies",
            "--match-id", match_id,
            "--video", str(video_file),
        ], cwd=ROOT)

        # run heatmaps
        run([
            "bash",
            str(ROOT / "heatmaps" / "run_heatmaps.bash"),
            match_id,
        ], cwd=ROOT)

        # delete clips after success
        if clips_dir.exists():
            print(f"Deleting temporary clips for {match_id}: {clips_dir}")
            shutil.rmtree(clips_dir)

        print(f"Done with {match_id}")

if __name__ == "__main__":
    main()
