import json
import subprocess
from pathlib import Path
import argparse


def cut_clip(start, end, output_path, video_path):
    duration = end - start

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        str(output_path)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    match_id = args.match_id
    video_path = args.video

    base = Path("rally_segmentator/output") / match_id
    input_json = base / "rallies_with_hitl.json"

    clips_dir = base / "rally_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    with open(input_json, "r") as f:
        rallies = json.load(f)

    print("Total rallies:", len(rallies))

    updated = []

    for i, r in enumerate(rallies):
        pad = 0.5

        start = max(0, r["start"] - pad)
        end = r["end"] + pad
        clip_name = f"rally_{i:03d}.mp4"
        clip_path = clips_dir / clip_name

        print(f"[{i}] {start:.2f} -> {end:.2f}")

        cut_clip(start, end, clip_path, video_path)

        r_new = r.copy()
        r_new["id"] = i
        r_new["clip_path"] = str(clip_path.relative_to(base))
        updated.append(r_new)

    output_json = base / "rallies_with_clips.json"

    with open(output_json, "w") as f:
        json.dump(updated, f, indent=2)

    print("\nDone")
    print("Saved:", output_json)


if __name__ == "__main__":
    main()
