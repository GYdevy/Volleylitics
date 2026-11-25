"""Cross-platform command-line runner for the Volleylitics pipeline.

The desktop GUI starts this module in a child process.  It is also useful on
its own for automation and prints structured stage events that the GUI can
turn into status cards.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Iterable

from config import BALL_MODEL, RALLY_MODEL, WHISTLE_MODEL


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent
EVENT_PREFIX = "@@VOLLEYLITICS@@"
EVENT_LOG_PATH: Path | None = None
HEATMAP_RESTART_EXIT_CODE = 75
REQUIRED_MODULES = (
    "numpy",
    "cv2",
    "librosa",
    "scipy",
    "torch",
    "torchvision",
    "tqdm",
    "ultralytics",
    "PIL",
)


def emit(stage: str, status: str, message: str = "", **extra) -> None:
    payload = {"stage": stage, "status": status, "message": message, **extra}
    serialized = json.dumps(payload, ensure_ascii=False)
    if EVENT_LOG_PATH is not None:
        with EVENT_LOG_PATH.open("a", encoding="utf-8") as event_log:
            event_log.write(serialized + "\n")
    print(f"{EVENT_PREFIX}{serialized}", flush=True)


def validate_match_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    if not cleaned:
        raise argparse.ArgumentTypeError("match id must contain letters or numbers")
    return cleaned


def parse_segments(value: str | None) -> list[list[float]] | None:
    if not value:
        return None
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid set ranges JSON: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("Set ranges must be a non-empty JSON list")
    normalized = []
    for index, pair in enumerate(data, 1):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Set range {index} must contain [start, end]")
        start, end = map(float, pair)
        if start < 0 or end <= start:
            raise ValueError(f"Set range {index} has invalid start/end values")
        normalized.append([start, end])
    return normalized


def validate_calibration(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read calibration file {path}: {exc}") from exc
    points = data.get("img_pts")
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("Calibration must contain four court corner points")
    for key in ("net_left_x", "net_right_x", "net_y_img"):
        if key not in data:
            raise ValueError(f"Calibration is missing {key}")
    if int(data["net_left_x"]) == int(data["net_right_x"]):
        raise ValueError("The two net calibration points cannot share the same x position")


def detect_compute_device(requested: str) -> tuple[str, str]:
    import torch

    if requested == "cpu":
        return "cpu", f"CPU · PyTorch {torch.__version__}"

    if not torch.cuda.is_available():
        if requested == "gpu":
            raise RuntimeError(
                "GPU acceleration was requested, but PyTorch cannot see a GPU. "
                "Run setup_windows.ps1 to install the AMD ROCm or NVIDIA CUDA build."
            )
        return "cpu", f"CPU fallback · PyTorch {torch.__version__}"

    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)
    backend = "AMD ROCm" if hip_version else "NVIDIA CUDA" if cuda_version else "GPU"
    backend_version = hip_version or cuda_version or "unknown"
    device_name = torch.cuda.get_device_name(0)

    try:
        test = torch.arange(16, device="cuda", dtype=torch.float32)
        result = (test * test).sum()
        torch.cuda.synchronize()
        if result.item() != 1240.0:
            raise RuntimeError("GPU calculation returned an unexpected result")
    except Exception as exc:
        raise RuntimeError(
            f"{backend} detects {device_name}, but a GPU kernel could not run: {exc}. "
            "If the AMD graphics driver was just installed or updated, restart Windows first."
        ) from exc

    return "cuda", (
        f"{backend} {backend_version} · {device_name} · PyTorch {torch.__version__}"
    )


def preflight(
    video: Path,
    calibration: Path | None,
    requested_device: str,
    whistle_model: Path,
    rally_model: Path,
    ball_model: Path,
) -> tuple[str, str]:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg was not found on PATH. Install it with 'winget install Gyan.FFmpeg'."
        )

    missing_modules = [
        name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None
    ]
    if missing_modules:
        raise RuntimeError(
            "Missing Python packages: "
            + ", ".join(missing_modules)
            + ". Run setup_windows.ps1 first."
        )

    required_models = [whistle_model, rally_model, ball_model]
    missing_models = [str(path) for path in required_models if not path.is_file()]
    if missing_models:
        raise FileNotFoundError(
            "Missing model file(s): "
            + ", ".join(missing_models)
            + ". See README.md for the expected filenames."
        )

    if calibration is None or not calibration.is_file():
        raise FileNotFoundError(
            "A saved court calibration is required for net-height ball detection"
        )
    validate_calibration(calibration)

    return detect_compute_device(requested_device)


def run_command(command: Iterable[str], log_file) -> None:
    command = [str(part) for part in command]
    command_display = subprocess.list2cmdline(command)
    print(f"> {command_display}", flush=True)
    log_file.write(f"> {command_display}\n")
    log_file.flush()

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    numba_cache = PROJECT_ROOT / "output" / ".cache" / "numba"
    numba_cache.mkdir(parents=True, exist_ok=True)
    environment.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
    miopen_db = PROJECT_ROOT / "output" / ".cache" / "miopen-db"
    miopen_cache = PROJECT_ROOT / "output" / ".cache" / "miopen-kernels"
    yolo_config = PROJECT_ROOT / "output" / ".cache" / "ultralytics"
    for cache_path in (miopen_db, miopen_cache, yolo_config):
        cache_path.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MIOPEN_USER_DB_PATH", str(miopen_db))
    environment.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(miopen_cache))
    environment.setdefault("YOLO_CONFIG_DIR", str(yolo_config))
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def save_pipeline_state(path: Path, phase: str, **details) -> None:
    path.write_text(
        json.dumps({"phase": phase, **details}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def wait_for_review_control(path: Path) -> str:
    while True:
        if path.is_file():
            decision = path.read_text(encoding="utf-8").strip().lower()
            path.unlink(missing_ok=True)
            if decision in {"continue", "reviewed", "pause"}:
                return decision
        time.sleep(0.25)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Volleylitics on one match video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--match-id", required=True, type=validate_match_id)
    parser.add_argument(
        "--calibration",
        required=True,
        help="Court calibration used for the net-height ball boundary",
    )
    parser.add_argument("--segments-json")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--event-log", help="Optional JSON-lines stage event file")
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="Compute device policy (default: auto)",
    )
    parser.add_argument("--skip-heatmap", action="store_true")
    parser.add_argument(
        "--heatmap-gpu-restarts",
        type=int,
        default=5,
        help="Fresh worker processes allowed after recoverable GPU launch failures",
    )
    parser.add_argument("--whistle-model", default=str(WHISTLE_MODEL))
    parser.add_argument("--rally-model", default=str(RALLY_MODEL))
    parser.add_argument("--ball-model", default=str(BALL_MODEL))
    parser.add_argument(
        "--review-mode",
        choices=("auto", "prompt"),
        default="auto",
        help="Pause after ambiguity detection for GUI review",
    )
    parser.add_argument("--review-control", help="GUI decision file for review-mode prompt")
    parser.add_argument(
        "--resume-after-review",
        action="store_true",
        help="Reuse saved rally detection and continue with clips/heatmaps",
    )
    parser.add_argument(
        "--resume-heatmap",
        action="store_true",
        help="Reuse saved rally clips and run only ball tracking/heatmap generation",
    )
    args = parser.parse_args()
    if args.heatmap_gpu_restarts < 0:
        parser.error("--heatmap-gpu-restarts cannot be negative")
    if args.resume_after_review and args.resume_heatmap:
        parser.error("--resume-after-review and --resume-heatmap cannot be combined")

    global EVENT_LOG_PATH
    if args.event_log:
        EVENT_LOG_PATH = Path(args.event_log).resolve()
        EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    video = Path(args.video).expanduser().resolve()
    calibration = Path(args.calibration).resolve() if args.calibration else None
    whistle_model = Path(args.whistle_model).expanduser().resolve()
    rally_model = Path(args.rally_model).expanduser().resolve()
    ball_model = Path(args.ball_model).expanduser().resolve()
    segments = parse_segments(args.segments_json)
    match_dir = Path(args.output_root).resolve() / args.match_id
    heatmap_dir = match_dir / "heatmap"
    match_dir.mkdir(parents=True, exist_ok=True)
    log_path = match_dir / "pipeline.log"
    state_path = match_dir / "pipeline_state.json"
    review_control = (
        Path(args.review_control).expanduser().resolve()
        if args.review_control
        else match_dir / "review-control.txt"
    )
    if not args.resume_after_review:
        review_control.unlink(missing_ok=True)

    resuming = args.resume_after_review or args.resume_heatmap
    log_mode = "a" if resuming else "w"
    with log_path.open(log_mode, encoding="utf-8", buffering=1) as log_file:
        if args.resume_after_review:
            log_file.write("\n=== Resuming after ambiguity review ===\n")
        elif args.resume_heatmap:
            log_file.write("\n=== Resuming ball tracking from saved clips ===\n")
        failure_phase = "failed"
        try:
            emit("preflight", "running", "Checking Windows runtime and project files")
            _torch_device, device_summary = preflight(
                video,
                calibration,
                args.device,
                whistle_model,
                rally_model,
                ball_model,
            )
            print(f"Whistle model: {whistle_model}")
            print(f"Rally model: {rally_model}")
            print(f"Ball model: {ball_model}")
            emit("preflight", "complete", device_summary)

            emit(
                "calibration",
                "complete",
                f"Using {calibration.name} for the net-height ball boundary",
            )

            rallies_path = match_dir / "rallies.json"
            final_rallies = match_dir / "rallies_with_hitl.json"
            hitl_path = match_dir / "hitl.json"

            if args.resume_heatmap:
                required_resume_files = [
                    rallies_path,
                    final_rallies,
                    match_dir / "rallies_with_clips.json",
                ]
                missing_resume_files = [
                    str(path) for path in required_resume_files if not path.is_file()
                ]
                if missing_resume_files or not (match_dir / "rally_clips").is_dir():
                    raise FileNotFoundError(
                        "Cannot resume ball tracking; missing saved clips or metadata: "
                        + ", ".join(missing_resume_files)
                    )
                rallies = json.loads(rallies_path.read_text(encoding="utf-8"))
                reviewed_rallies = json.loads(final_rallies.read_text(encoding="utf-8"))
                emit("rallies", "complete", f"Reusing {len(rallies)} detected rallies")
                emit("review", "complete", "Using saved ambiguity-review decisions")
                emit("clips", "complete", f"Reusing {len(reviewed_rallies)} rally clips")
            elif args.resume_after_review:
                if not rallies_path.is_file() or not hitl_path.is_file():
                    raise FileNotFoundError(
                        "Cannot resume: rallies.json and hitl.json must already exist"
                    )
                if not final_rallies.is_file():
                    shutil.copyfile(rallies_path, final_rallies)
                rallies = json.loads(rallies_path.read_text(encoding="utf-8"))
                emit("rallies", "complete", f"Reusing {len(rallies)} detected rallies")
                emit("review", "complete", "Using saved ambiguity-review decisions")
            else:
                save_pipeline_state(state_path, "detecting_rallies", video=str(video))
                emit("rallies", "running", "Detecting whistles and rally intervals")
                timeline_command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "rally_segmentator.rally_timeline_maker",
                    "--match-id",
                    args.match_id,
                    "--video",
                    video,
                    "--output-dir",
                    match_dir,
                    "--calibration",
                    calibration,
                    "--device",
                    args.device,
                    "--whistle-model",
                    whistle_model,
                    "--rally-model",
                    rally_model,
                    "--ball-model",
                    ball_model,
                ]
                if segments:
                    timeline_command.extend(["--segments-json", json.dumps(segments)])
                run_command(timeline_command, log_file)
                rallies = json.loads(rallies_path.read_text(encoding="utf-8"))
                shutil.copyfile(rallies_path, final_rallies)
                emit("rallies", "complete", f"Detected {len(rallies)} rallies")

            if not args.resume_heatmap:
                hitl = json.loads(hitl_path.read_text(encoding="utf-8"))
                if (
                    not args.resume_after_review
                    and args.review_mode == "prompt"
                    and hitl
                ):
                    save_pipeline_state(
                        state_path,
                        "awaiting_review",
                        video=str(video),
                        match_id=args.match_id,
                        hitl_count=len(hitl),
                    )
                    emit(
                        "review",
                        "waiting",
                        f"{len(hitl)} ambiguous intervals need a decision",
                        hitl_count=len(hitl),
                        control_file=str(review_control),
                        output_dir=str(match_dir),
                    )
                    review_decision = wait_for_review_control(review_control)
                    if review_decision == "pause":
                        emit(
                            "complete",
                            "paused",
                            "Paused before clips; import this folder to review or resume later",
                            output_dir=str(match_dir),
                        )
                        return 2
                    if review_decision == "reviewed":
                        emit("review", "complete", "Manual ambiguity decisions integrated")
                    else:
                        emit("review", "skipped", "Continued without manual ambiguity review")
                elif not args.resume_after_review:
                    review_message = (
                        f"Kept {len(rallies)} automatic rallies; "
                        f"saved {len(hitl)} uncertain intervals"
                    )
                    emit("review", "complete", review_message)

                reviewed_rallies = json.loads(final_rallies.read_text(encoding="utf-8"))
                save_pipeline_state(
                    state_path,
                    "creating_clips",
                    video=str(video),
                    rally_count=len(reviewed_rallies),
                )

                emit("clips", "running", "Creating one clip for each rally")
                run_command(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "rally_segmentator.split_rallies",
                        "--match-id",
                        args.match_id,
                        "--video",
                        video,
                        "--output-dir",
                        match_dir,
                        "--rallies-json",
                        final_rallies,
                    ],
                    log_file,
                )
                emit("clips", "complete", f"Created {len(reviewed_rallies)} rally clips")

            if args.skip_heatmap:
                emit("heatmap", "skipped", "Heatmap generation was disabled")
            elif not reviewed_rallies:
                emit("heatmap", "skipped", "No rallies were available for ball tracking")
            else:
                failure_phase = "heatmap_failed"
                save_pipeline_state(
                    state_path,
                    "tracking_ball",
                    video=str(video),
                    rally_count=len(reviewed_rallies),
                )
                emit("heatmap", "running", "Tracking ball landings and drawing the court map")
                heatmap_command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "heatmaps.generate_heatmap",
                    "--match-id",
                    args.match_id,
                    "--calibration",
                    calibration,
                    "--clips-dir",
                    match_dir / "rally_clips",
                    "--rallies-json",
                    match_dir / "rallies_with_clips.json",
                    "--output-dir",
                    heatmap_dir,
                    "--device",
                    args.device,
                    "--ball-model",
                    ball_model,
                ]
                for restart_number in range(args.heatmap_gpu_restarts + 1):
                    try:
                        run_command(heatmap_command, log_file)
                        break
                    except subprocess.CalledProcessError as exc:
                        can_restart = (
                            exc.returncode == HEATMAP_RESTART_EXIT_CODE
                            and restart_number < args.heatmap_gpu_restarts
                        )
                        if not can_restart:
                            raise
                        completed_restarts = restart_number + 1
                        message = (
                            "AMD/NVIDIA worker reset detected; restarting ball tracking "
                            f"from its checkpoint ({completed_restarts}/"
                            f"{args.heatmap_gpu_restarts})"
                        )
                        print(message, flush=True)
                        log_file.write(message + "\n")
                        log_file.flush()
                        emit("heatmap", "running", message)
                emit("heatmap", "complete", "Court map and rally analytics are ready")

            emit(
                "complete",
                "complete",
                f"Processing finished. Results: {match_dir}",
                output_dir=str(match_dir),
            )
            save_pipeline_state(
                state_path,
                "complete",
                video=str(video),
                rally_count=len(reviewed_rallies),
            )
            return 0
        except Exception as exc:
            save_pipeline_state(
                state_path,
                failure_phase,
                video=str(video),
                error=f"{type(exc).__name__}: {exc}",
            )
            log_file.write(f"ERROR: {type(exc).__name__}: {exc}\n")
            emit("complete", "failed", f"{type(exc).__name__}: {exc}")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
