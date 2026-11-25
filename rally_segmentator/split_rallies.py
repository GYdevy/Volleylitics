import json
import shutil
import subprocess
from pathlib import Path
import argparse
import os


def choose_video_encoder(ffmpeg):
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    available = result.stdout
    if os.environ.get("HIP_PATH") and "h264_amf" in available:
        return ["-c:v", "h264_amf", "-quality", "speed"]
    if shutil.which("nvidia-smi") and "h264_nvenc" in available:
        return ["-c:v", "h264_nvenc", "-preset", "p1"]
    if "libx264" in available:
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22"]
    if os.name == "nt" and "h264_mf" in available:
        return ["-c:v", "h264_mf"]
    return ["-c:v", "mpeg4", "-q:v", "3"]


def cut_clip(start, end, output_path, video_path, ffmpeg, video_encoder):
    duration = end - start

    cmd = [
        ffmpeg,
        "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-map", "0:v:0",
        "-map", "0:a?",
        *video_encoder,
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def main():
    parser = argparse.ArgumentParser(description="Split detected rallies into clips")
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--output-dir",
        help="Match output directory (defaults to rally_segmentator/output/<match-id>)",
    )
    parser.add_argument(
        "--rallies-json",
        help="Rally JSON to split (defaults to rallies_with_hitl.json, then rallies.json)",
    )
    args = parser.parse_args()

    match_id = args.match_id
    video_path = args.video

    base = (
        Path(args.output_dir)
        if args.output_dir
        else Path("rally_segmentator/output") / match_id
    )
    if args.rallies_json:
        input_json = Path(args.rallies_json)
    else:
        input_json = base / "rallies_with_hitl.json"
        if not input_json.is_file():
            input_json = base / "rallies.json"

    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not input_json.is_file():
        raise FileNotFoundError(f"Rally timeline not found: {input_json}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required but was not found on PATH")
    video_encoder = choose_video_encoder(ffmpeg)
    print("Clip encoder:", " ".join(video_encoder))

    clips_dir = base / "rally_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        rallies = json.load(f)

    rallies = sorted(rallies, key=lambda rally: (rally["start"], rally["end"]))
    for previous, current in zip(rallies, rallies[1:]):
        if current["start"] < previous["end"] - 1e-6:
            raise ValueError(
                "Rally intervals overlap before clipping: "
                f"{previous['start']:.2f}-{previous['end']:.2f} and "
                f"{current['start']:.2f}-{current['end']:.2f}"
            )

    print("Total rallies:", len(rallies))

    updated = []
    current_clip_names = set()

    for i, r in enumerate(rallies):
        pad = 0.5

        start = max(0, r["start"] - pad)
        end = r["end"] + pad
        if i > 0:
            previous_boundary = (rallies[i - 1]["end"] + r["start"]) / 2
            start = max(start, previous_boundary)
        if i + 1 < len(rallies):
            next_boundary = (r["end"] + rallies[i + 1]["start"]) / 2
            end = min(end, next_boundary)
        if end <= start:
            raise ValueError(f"Invalid clip range after overlap protection: {start} - {end}")
        clip_name = f"rally_{i:03d}.mp4"
        clip_path = clips_dir / clip_name
        current_clip_names.add(clip_name)

        print(f"[{i}] {start:.2f} -> {end:.2f}")

        cut_clip(start, end, clip_path, video_path, ffmpeg, video_encoder)

        r_new = r.copy()
        r_new["id"] = i
        r_new["clip_start"] = start
        r_new["clip_end"] = end
        r_new["clip_path"] = clip_path.relative_to(base).as_posix()
        updated.append(r_new)

    # Remove only obsolete generated clips.  Without this, rerunning a match
    # with fewer rallies leaves old rally_*.mp4 files that look like duplicates.
    for old_clip in clips_dir.glob("rally_*.mp4"):
        if old_clip.name not in current_clip_names:
            old_clip.unlink()

    output_json = base / "rallies_with_clips.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)

    print("\nDone")
    print("Saved:", output_json)


if __name__ == "__main__":
    main()
