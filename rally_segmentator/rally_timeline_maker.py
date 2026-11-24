import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import os
import argparse
import time
import shutil
import subprocess
from ultralytics import YOLO
from config import (
    BALL_MODEL,
    WHISTLE_MODEL,
    RALLY_MODEL,
    VIDEO_DIR,
    RALLY_BASE,
)

from rally_segmentator.dsp_detector import (
    load_audio_from_video,
    detect_active_frames,
    group_frames,
    extract_candidates,
    refine_candidates,
    estimate_whistle_band,
    compute_match_stats,
    rule_based_sifter,
    cfg
)

from rally_segmentator.models import (
    ResNetWhistle,
    build_rally_model,
    wav_to_logmel_from_audio,
)


# =============================
# SETTINGS
# =============================


SEGMENTS = [(0.0, float("inf"))]

DEVICE = "cpu"

SNIPPET_SEC = 1.0
HALF_SNIPPET = SNIPPET_SEC / 2

ENTER = 0.65
EXIT = 0.4
MIN_RALLY_RATIO = 0.35
FP_RATIO = 0.1

TIMELINE_STEP = 0.2


MAX_RALLY_DURATION = 35

# =============================
# LOAD MODELS
# =============================

def load_models(whistle_model_path, rally_model_path, ball_model_path):
    print(f"Loading models on {DEVICE}...")

    for model_path in (whistle_model_path, rally_model_path, ball_model_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Required model not found: {model_path}")

    whistle_model = ResNetWhistle().to(DEVICE)
    checkpoint = torch.load(whistle_model_path, map_location=DEVICE)
    whistle_model.load_state_dict(checkpoint["model_state"])
    whistle_model.eval()

    rally_model = build_rally_model().to(DEVICE)
    rally_model.load_state_dict(
        torch.load(rally_model_path, map_location=DEVICE)
    )
    rally_model.eval()

    ball_model = YOLO(str(ball_model_path))

    return whistle_model, rally_model, ball_model


def select_device(policy):
    if policy == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if policy == "gpu":
        raise RuntimeError("GPU processing was requested but no GPU backend is available")
    return "cpu"


def default_batch_size():
    """Choose a conservative inference batch from the available GPU memory."""

    if DEVICE != "cuda":
        return 16
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if memory_gb >= 12:
        return 128
    if memory_gb >= 6:
        return 64
    return 32




# =============================
# HELPERS
# =============================

def interval_ball_activity(start, end, ball_t, ball_confidences):
    mask = (ball_t >= start) & (ball_t <= end)
    confidences = ball_confidences[mask]
    if confidences.size == 0:
        return 0.0, 0, 0.0
    active = confidences > 0
    hits = int(np.count_nonzero(active))
    ratio = hits / len(confidences)
    mean_confidence = float(np.mean(confidences[active])) if hits else 0.0
    return float(ratio), hits, mean_confidence


def hysteresis(probs):
    state = 0
    out = []

    for p in probs:
        if state == 0 and p > ENTER:
            state = 1
        elif state == 1 and p < EXIT:
            state = 0

        out.append(state)

    return np.array(out)

def remove_short_runs(states, min_len=3):
    out = states.copy()
    i = 0

    while i < len(states):
        j = i
        while j < len(states) and states[j] == states[i]:
            j += 1

        if (j - i) < min_len:
            out[i:j] = 1 - states[i]

        i = j

    return out

def in_segments(t):
    for s,e in SEGMENTS:
        if s <= t <= e:
            return True
    return False

def smooth_signal(x, k=7):
    out = np.zeros_like(x)
    half = k // 2

    for i in range(len(x)):
        s = max(0, i - half)
        e = min(len(x), i + half + 1)
        out[i] = np.mean(x[s:e])

    return out


def window_nms(candidates, window=1.5):
    # Confidence-first NMS guarantees that two returned whistle boundaries are
    # never closer than `window`.  The previous window partition could pick a
    # point at the end of one cluster and another at the start of the next,
    # producing sub-second partial clips.
    selected = []
    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        timestamp, _probability = candidate
        if all(abs(timestamp - kept[0]) > window for kept in selected):
            selected.append(candidate)
    return [timestamp for timestamp, _probability in sorted(selected)]


def merge_consecutive_rallies(rallies):
    """Merge adjacent strong intervals split by a false whistle boundary."""

    merged = []
    for rally in rallies:
        current = rally.copy()
        current["merged_intervals"] = 1
        strong_pair = (
            merged
            and current["set"] == merged[-1]["set"]
            and current["_interval_index"] == merged[-1]["_last_interval_index"] + 1
            and current["_strong_cnn"]
            and merged[-1]["_strong_cnn"]
        )
        short_supported_continuation = (
            merged
            and current["set"] == merged[-1]["set"]
            and current["_interval_index"] == merged[-1]["_last_interval_index"] + 1
            and (
                (
                    merged[-1]["_strong_cnn"]
                    and not current["_strong_cnn"]
                    and current["duration"] <= 6.0
                    and current["_ball_ratio"] >= 0.15
                )
                or (
                    current["_strong_cnn"]
                    and not merged[-1]["_strong_cnn"]
                    and merged[-1]["duration"] <= 6.0
                    and merged[-1]["_ball_ratio"] >= 0.15
                )
            )
        )
        if strong_pair or short_supported_continuation:
            previous = merged[-1]
            previous["end"] = current["end"]
            previous["duration"] = previous["end"] - previous["start"]
            previous["label"] = "CNN_MERGED"
            previous["merged_intervals"] += 1
            previous["_last_interval_index"] = current["_interval_index"]
            previous["_strong_cnn"] = (
                previous["_strong_cnn"] or current["_strong_cnn"]
            )
            previous["_ball_ratio"] = max(
                previous["_ball_ratio"], current["_ball_ratio"]
            )
            continue

        current["_last_interval_index"] = current["_interval_index"]
        merged.append(current)

    for rally in merged:
        rally.pop("_interval_index", None)
        rally.pop("_last_interval_index", None)
        rally.pop("_strong_cnn", None)
        rally.pop("_ball_ratio", None)
        if rally.get("merged_intervals") == 1:
            rally.pop("merged_intervals", None)
    return merged

# =============================
# DSP WHISTLES
# =============================

def detect_whistles_dsp(y):
    active = detect_active_frames(y)
    groups = group_frames(active)
    stage1 = extract_candidates(groups)

    low, high = estimate_whistle_band(y, stage1)
    cfg.whistle_low = low
    cfg.whistle_high = high

    active = detect_active_frames(y)
    groups = group_frames(active)
    stage1 = extract_candidates(groups)

    refined = refine_candidates(y, stage1)
    stats = compute_match_stats(refined, y)

    return rule_based_sifter(refined, y, stats)


# =============================
# CNN FILTER
# =============================

def filter_whistles_cnn(accepted, y, model, batch_size):
    cnn_candidates = []
    pending_mels = []
    pending_centers = []

    def flush_batch():
        if not pending_mels:
            return
        batch = torch.stack(pending_mels).to(DEVICE, non_blocking=DEVICE == "cuda")
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(batch)).reshape(-1).cpu().numpy()
        cnn_candidates.extend(
            (center, float(probability))
            for center, probability in zip(pending_centers, probabilities)
            if probability > 0.08
        )
        pending_mels.clear()
        pending_centers.clear()

    for start, end in tqdm(accepted, desc="Whistle CNN"):
        center = (start + end) / 2

        s0 = int((center - HALF_SNIPPET) * cfg.sr)
        s1 = int((center + HALF_SNIPPET) * cfg.sr)

        if s0 < 0 or s1 > len(y):
            continue

        snippet = y[s0:s1]
        mel = wav_to_logmel_from_audio(snippet)
        pending_mels.append(torch.from_numpy(mel))
        pending_centers.append(center)
        if len(pending_mels) >= batch_size:
            flush_batch()

    flush_batch()

    return cnn_candidates


# =============================
# TIMELINE
# =============================

def build_timeline(model, ball_model, batch_size, net_y_img):
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or source_height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video metadata: {VIDEO_PATH}")
    duration = frame_count / fps
    cap.release()

    decode_fps = int(round(1 / TIMELINE_STEP))
    frame_width, frame_height = 960, 540
    frame_bytes = frame_width * frame_height * 3
    total_steps = int(np.ceil(duration / TIMELINE_STEP))
    pbar = tqdm(total=total_steps, desc="Video models")
    scaled_net_y = float(net_y_img) * frame_height / source_height
    print(
        f"Ball activity boundary: source y={net_y_img}, "
        f"inference y={scaled_net_y:.1f}"
    )

    timeline_t = []
    timeline_p = []
    ball_t = []
    ball_confidences = []
    pending_rally_frames = []
    pending_rally_times = []
    pending_ball_frames = []
    pending_ball_times = []
    ball_batch_size = 32 if DEVICE == "cuda" else 8
    ball_device = 0 if DEVICE == "cuda" else "cpu"

    def flush_rally_batch():
        if not pending_rally_frames:
            return
        batch = torch.stack(pending_rally_frames).to(
            DEVICE, non_blocking=DEVICE == "cuda"
        )
        batch = batch.float().div_(255.0)
        with torch.inference_mode():
            probabilities = F.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
        timeline_t.extend(pending_rally_times)
        timeline_p.extend(float(value) for value in probabilities)
        pbar.update(len(pending_rally_frames))
        pending_rally_frames.clear()
        pending_rally_times.clear()

    def flush_ball_batch():
        if not pending_ball_frames:
            return
        results = ball_model.predict(
            source=pending_ball_frames,
            imgsz=960,
            conf=0.10,
            device=ball_device,
            batch=ball_batch_size,
            verbose=False,
        )
        for timestamp, result in zip(pending_ball_times, results):
            active_confidence = 0.0
            boxes = result.boxes
            if len(boxes):
                centers_y = (boxes.xyxy[:, 1] + boxes.xyxy[:, 3]) / 2
                above_net = centers_y <= scaled_net_y
                if bool(above_net.any()):
                    active_confidence = float(boxes.conf[above_net].max().item())
            ball_t.append(timestamp)
            ball_confidences.append(active_confidence)
        pending_ball_frames.clear()
        pending_ball_times.clear()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for the optimized video scan")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", VIDEO_PATH,
        "-an",
        "-vf", f"fps={decode_fps},scale={frame_width}:{frame_height}:flags=bilinear",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creation_flags,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    frame_idx = 0

    try:
        while True:
            raw_frame = process.stdout.read(frame_bytes)
            if not raw_frame:
                break
            if len(raw_frame) != frame_bytes:
                raise RuntimeError("FFmpeg returned an incomplete video frame")
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                frame_height, frame_width, 3
            )
            timestamp = frame_idx / decode_fps

            model_frame = cv2.resize(
                frame, (224, 224), interpolation=cv2.INTER_AREA
            )
            model_frame = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)
            pending_rally_frames.append(
                torch.from_numpy(model_frame).permute(2, 0, 1).contiguous()
            )
            pending_rally_times.append(timestamp)
            pending_ball_frames.append(frame.copy())
            pending_ball_times.append(timestamp)
            if len(pending_rally_frames) >= batch_size:
                flush_rally_batch()
            if len(pending_ball_frames) >= ball_batch_size:
                flush_ball_batch()

            frame_idx += 1

        flush_rally_batch()
        flush_ball_batch()
        stderr_text = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f"FFmpeg video scan failed ({return_code}): {stderr_text or 'unknown error'}"
            )
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        pbar.close()

    return (
        np.asarray(timeline_t),
        np.asarray(timeline_p),
        np.asarray(ball_t),
        np.asarray(ball_confidences),
    )

# =============================
# ANALYSIS
# =============================
def get_set_for_time(t, segments):
    for i, (start, end) in enumerate(segments):
        if start <= t <= end:
            return i + 1
    return None

def analyze_intervals(
    cnn_centers, timeline_t, timeline_p, ball_t, ball_confidences
):
    rallies = []
    hitl_candidates = []
    timeline_p_smooth = smooth_signal(timeline_p, k=9)
    for i in tqdm(range(len(cnn_centers) - 1), desc="Analyzing intervals"):

        w0 = cnn_centers[i]
        w1 = cnn_centers[i + 1]
        set_id = get_set_for_time(w0, SEGMENTS)
        if not in_segments(w0):
            continue

        mask = (timeline_t >= w0) & (timeline_t <= w1)
        probs = timeline_p_smooth[mask]

        if len(probs) == 0:
            continue

        # CNN-based signal
        states = hysteresis(probs)
        states = remove_short_runs(states, min_len=3)
        inplay_ratio = np.mean(states)

        # A detection only counts as active when its center is above the
        # calibrated net-height boundary.
        ball_ratio, ball_hits, ball_confidence = interval_ball_activity(
            w0, w1, ball_t, ball_confidences
        )

        
        score = inplay_ratio + 0.75 * ball_ratio + 0.10 * ball_confidence
        duration = w1 - w0

        if inplay_ratio > MIN_RALLY_RATIO:
            reason = "CNN"
            is_rally = True

    # The ball model supports a borderline rally-model decision, but never
    # creates a rally from an otherwise inactive interval.  This prevents a
    # serve toss between the end whistle and next service whistle from becoming
    # a duplicate rally.
        elif (
            inplay_ratio > 0.25
            and ball_hits >= 3
            and ball_ratio >= 0.08
        ):
            reason = "BALL_SUPPORTED"
            is_rally = True

        elif inplay_ratio > 0.20 and ball_hits >= 3 and ball_ratio >= 0.08:
            reason = "BALL_SUPPORTED"
            is_rally = True

        elif (
            inplay_ratio > 0.15
            and ball_hits >= 5
            and ball_ratio >= 0.20
        ):
            reason = "BALL_SUPPORTED_WEAK"
            is_rally = True

    # Ball-only and other ambiguous intervals are review candidates, not clips.
        elif inplay_ratio >= 0.10 or (ball_hits >= 3 and ball_ratio >= 0.04):
            reason = "HITL_BALL_ONLY" if inplay_ratio < 0.10 else "HITL"
            is_rally = False
            hitl_candidates.append({
            "start": float(w0),
            "end": float(w1),
            "duration": float(duration),
            "ratio": float(inplay_ratio),
            "ball_ratio": float(ball_ratio),
            "ball_hits": int(ball_hits),
            "ball_confidence": float(ball_confidence),
            "score": float(score),
            "index": i,
            "set": set_id
        })

    # REJECT
        else:
            reason = "REJECT"
            is_rally = False 

        print(
            f"{i}: start:{w0}, end:{w1} inplay_ratio={inplay_ratio:.2f}, "
            f"ball_ratio={ball_ratio:.2f}, ball_hits={ball_hits}, "
            f"ball_conf={ball_confidence:.2f}, dur={duration:.2f}, "
            f"score={score:.2f} → {reason}"
        )

        # =========================
        # APPLY DECISION
        # =========================

        if is_rally:

            rallies.append({
                "start": float(w0),
                "end": float(w1),
                "duration": float(duration),
                "label": reason,
                "set": set_id,
                "_interval_index": i,
                "_strong_cnn": reason == "CNN",
                "_ball_ratio": float(ball_ratio),
            })

            if duration > MAX_RALLY_DURATION:
                print("Possible missed whistle near", w0)

        elif inplay_ratio < FP_RATIO:
            print("Possible FP whistle between", w0, w1)
    return merge_consecutive_rallies(rallies), hitl_candidates


    
# =============================
# SAVE
# =============================
def save_results(rallies, hitl, match_id, output_dir=None):
    match_output_dir = (
        Path(output_dir) if output_dir else RALLY_BASE / "output" / match_id
    )
    match_output_dir.mkdir(parents=True, exist_ok=True)

    rallies_path = match_output_dir / "rallies.json"
    hitl_path = match_output_dir / "hitl.json"

    with open(rallies_path, "w") as f:
        json.dump(rallies, f, indent=2)

    with open(hitl_path, "w") as f:
        json.dump(hitl, f, indent=2)

    print(f"Rallies saved to: {rallies_path}")
    print(f"HITL saved to: {hitl_path}")


# =============================
# MAIN
# =============================

def main():
    parser = argparse.ArgumentParser(description="Detect volleyball rallies in a video")
    parser.add_argument("--match-id", required=True)
    parser.add_argument(
        "--video",
        help="Source video path (defaults to videos/<match-id>.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for rallies.json and hitl.json",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="Court calibration JSON containing net_y_img",
    )
    parser.add_argument(
        "--segments-json",
        help="JSON list of set ranges in seconds, e.g. [[0,600],[700,1300]]",
    )
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--whistle-model", default=str(WHISTLE_MODEL))
    parser.add_argument("--rally-model", default=str(RALLY_MODEL))
    parser.add_argument("--ball-model", default=str(BALL_MODEL))
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Inference batch size (auto-selected from GPU memory by default)",
    )
    args = parser.parse_args()

    match_id = args.match_id

    global VIDEO_PATH, SEGMENTS, DEVICE
    VIDEO_PATH = str(Path(args.video) if args.video else VIDEO_DIR / f"{match_id}.mp4")
    whistle_model_path = Path(args.whistle_model).expanduser().resolve()
    rally_model_path = Path(args.rally_model).expanduser().resolve()
    ball_model_path = Path(args.ball_model).expanduser().resolve()
    DEVICE = select_device(args.device)
    batch_size = args.batch_size or default_batch_size()
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    segments_json = args.segments_json or os.environ.get("SEGMENTS_JSON")
    if segments_json:
        SEGMENTS = json.loads(segments_json)
    if not SEGMENTS:
        SEGMENTS = [(0.0, float("inf"))]

    if not Path(VIDEO_PATH).is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    calibration_path = Path(args.calibration)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration not found: {calibration_path}")
    with calibration_path.open("r", encoding="utf-8") as calibration_file:
        calibration = json.load(calibration_file)
    if "net_y_img" not in calibration:
        raise ValueError("Calibration is missing net_y_img")
    net_y_img = float(calibration["net_y_img"])

    print("Running in:", os.getcwd())
    print(f"Analyzing {match_id}")
    print(f"Inference batch size: {batch_size}")

    print(f"Whistle model: {whistle_model_path}")
    print(f"Rally model: {rally_model_path}")
    print(f"Ball model: {ball_model_path}")
    whistle_model, rally_model, ball_model = load_models(
        whistle_model_path, rally_model_path, ball_model_path
    )

    print("Loading audio...")
    stage_started = time.perf_counter()
    y = load_audio_from_video(VIDEO_PATH, cfg.sr)
    print(f"Audio loaded in {time.perf_counter() - stage_started:.1f}s")

    print("DSP detection...")
    stage_started = time.perf_counter()
    accepted = detect_whistles_dsp(y)
    print("Accepted candidates:", len(accepted))
    print(f"DSP detection completed in {time.perf_counter() - stage_started:.1f}s")

    print("CNN filtering...")
    stage_started = time.perf_counter()
    cnn_centers = filter_whistles_cnn(accepted, y, whistle_model, batch_size)
    print("Total whistles:", len(cnn_centers))
    print(f"Whistle CNN completed in {time.perf_counter() - stage_started:.1f}s")

    print("Applying NMS for whistles...")
    cnn_centers = window_nms(cnn_centers, window=1.5)
    print("Total whistles after NMS:", len(cnn_centers))

    print("Building timeline...")
    stage_started = time.perf_counter()
    timeline_t, timeline_p, ball_t, ball_confidences = build_timeline(
        rally_model, ball_model, batch_size, net_y_img
    )
    print(f"Timeline completed in {time.perf_counter() - stage_started:.1f}s")

    print("Analyzing...")
    rallies, hitl = analyze_intervals(
        cnn_centers, timeline_t, timeline_p, ball_t, ball_confidences
    )

    save_results(rallies, hitl, match_id, args.output_dir)
    print("Rallies Count:", len(rallies))
    print("HITL Count:", len(hitl))
   
    

if __name__ == "__main__":
    main()
