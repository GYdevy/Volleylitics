import numpy as np
import librosa
import subprocess
from pathlib import Path
import csv
import pandas as pd
import joblib
import os
import shutil
import warnings

# ===============================
# SILENCE KNOWN HARMLESS WARNINGS
# ===============================
warnings.filterwarnings(
    "ignore",
    message="n_fft=.* is too large for input signal"
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

# ===============================
# CONFIG
# ===============================
MATCH_NUM = 4

BASE_VIDEO_DIR = r"D:\Volleyballey\videos"
BASE_OUTPUT_DIR = r"D:\Volleyballey\WhistleDetector"
VIDEO_PATH = fr"{BASE_VIDEO_DIR}\match{MATCH_NUM}.mp4"

OUTPUT_DIR = Path(fr"{BASE_OUTPUT_DIR}\clips\match{MATCH_NUM}")
FINAL_OUTPUT = fr"{BASE_OUTPUT_DIR}\compiled_matches\whistles_match{MATCH_NUM}.mp4"

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

MODEL_PATH = fr"{BASE_OUTPUT_DIR}\best_model.pkl"
AMBIG_MODEL_PATH = fr"{BASE_OUTPUT_DIR}\ambiguous_best_model.pkl"

AMBIG_DIR = Path(fr"{BASE_OUTPUT_DIR}\ambiguous\match{MATCH_NUM}")
TRAINING_CSV = Path(fr"{BASE_OUTPUT_DIR}\training_with_uid.csv")

# ===============================
# AUDIO CONFIG
# ===============================
SR = 22050
N_FFT = 2048
HOP = 128

WHISTLE_LOW = 3700
WHISTLE_HIGH = 4200

PAD_BEFORE = 0.10
PAD_AFTER  = 0.1

MIN_DURATION_SEC = 0.06
MIN_FRAMES = int(MIN_DURATION_SEC / (HOP / SR))

# ===============================
# THRESHOLDS
# ===============================
ENERGY_START_PCTL = 50
ENERGY_CONT_PCTL  = 30
MAX_GAP_FRAMES    = 4
HOLD_TIME_SEC     = 0.2

MIN_RIDGE  = 11
MIN_GRAD_F = 4.7

MAX_FLATNESS = 0.025
MAX_CENTROID = 3000
BANDWIDTH_MAX = 180.0  # Hz (tune later)
SCORE_OK = -24.0
CLS_THRESHOLD = 0.72
AMBIG_MODEL_LOW = 0.10
PHYSICS_MIN_PROBA = 0.02
PEAK_STD_MAX = 120.0   # Hz — whistles usually < 80, chants >> 150


# ===============================
# LOAD MODELS
# ===============================
clf = joblib.load(MODEL_PATH)
ambig_clf = joblib.load(AMBIG_MODEL_PATH)

# ===============================
# HELPERS
# ===============================
def fmt_time(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"

def band_width_hz(S_w, freqs_w):
    """
    Width (Hz) of concentrated energy inside whistle band.
    Uses only whistle-band frequencies.
    """
    if S_w.size == 0:
        return np.inf

    mean_spec = np.mean(S_w, axis=1)

    if mean_spec.max() <= 0:
        return np.inf

    active = mean_spec > (mean_spec.max() - 6)  # within 6 dB of peak

    if not np.any(active):
        return np.inf

    active_freqs = freqs_w[active]
    return active_freqs.max() - active_freqs.min()

def is_tonal_frame(spec, max_bins=6):
    peak = spec.max()
    if peak < -60:
        return False
    active_bins = np.sum(spec > (peak - 6))
    return active_bins <= max_bins

def split_by_tonality(group_frames, S_w, min_frames, max_gap=3):
    """
    Split a large energy group into tonal subgroups (candidate whistles).
    """
    tonal = []

    for f in group_frames:
        if is_tonal_frame(S_w[:, f]):
            tonal.append(f)
        else:
            tonal.append(None)

    subgroups = []
    cur = []

    for f in tonal:
        if f is not None:
            if not cur or f - cur[-1] <= max_gap:
                cur.append(f)
            else:
                if len(cur) >= min_frames:
                    subgroups.append(cur)
                cur = [f]
        else:
            if cur and len(cur) >= min_frames:
                subgroups.append(cur)
            cur = []

    if cur and len(cur) >= min_frames:
        subgroups.append(cur)

    return subgroups


def save_clip(start, end, folder):
    folder.mkdir(parents=True, exist_ok=True)
    safe_start = max(0, start - PAD_BEFORE)
    duration = (end - start) + PAD_BEFORE + PAD_AFTER

    out = folder / f"whistle_{int(start*1000):010d}_{int(end*1000):010d}.mp4"

    subprocess.run([
        FFMPEG, "-y",
        "-ss", str(safe_start),
        "-i", VIDEO_PATH,
        "-t", str(duration),
        "-c:v", "mpeg4", "-qscale:v", "3",
        "-c:a", "aac",
        str(out)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return out
def peak_freq_std(S_w, freqs):
    """
    Std of peak frequency over time (Hz).
    Low for whistles, high for voices/chants.
    """
    if S_w.shape[1] < 3:
        return np.inf

    peak_bins = np.argmax(S_w, axis=0)
    peak_freqs = freqs[peak_bins]
    return np.std(peak_freqs)

def physics_suspect(feats):
    score = 0

    if feats["flatness"] > 0.030:
        score += 1

    if feats["ridge"] < 4:
        score += 1

    if feats["grad_f"] < 4.3:
        score += 1

    return score >= 2  # ← key change

def append_training_row(uid, X, label):
    header = [
        "uid",
        "rms","flatness","centroid","band_energy","peak_freq",
        "ridge","grad_t","grad_f",
        "rolloff","bandwidth","zcr","contrast","tonnetz",
        "mfcc1","mfcc2","mfcc3","mfcc4","mfcc5",
        "label"
    ]

    file_exists = TRAINING_CSV.exists()

    if file_exists:
        existing = set(pd.read_csv(TRAINING_CSV)["uid"])
        if uid in existing:
            print(f"⚠ UID already exists, skipping {uid}")
            return

    with open(TRAINING_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([uid, *X.tolist(), label])


# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(y):
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))

    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    freqs = librosa.fft_frequencies(sr=SR)
    mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)
    S_w = S_db[mask]

    rms = librosa.feature.rms(y=y).mean()
    flat = librosa.feature.spectral_flatness(y=y).mean()
    cent = librosa.feature.spectral_centroid(y=y, sr=SR).mean()
    band_E = S_w.mean()
    peak_f = freqs[mask][np.argmax(S_w.mean(axis=1))]

    fe = np.percentile(S_w, 75, axis=0)
    ridge = np.sum(fe > (fe.mean() + fe.std()))
    grad_t = np.abs(np.diff(S_w, axis=1)).mean()
    grad_f = np.abs(np.diff(S_w, axis=0)).mean()

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR).mean()
    bw = librosa.feature.spectral_bandwidth(y=y, sr=SR).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    contrast = librosa.feature.spectral_contrast(S=S_mag, sr=SR).mean()
    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y), sr=SR
    ).mean()

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=5).mean(axis=1)

    physics = {
        "flatness": flat,
        "ridge": ridge,
        "grad_f": grad_f,
        "centroid": cent,
        "peak_freq": peak_f,
    }

    X = np.array([
        rms, flat, cent, band_E, peak_f,
        ridge, grad_t, grad_f,
        rolloff, bw, zcr, contrast, tonnetz,
        *mfcc
    ])

    return physics, X

# ===============================
# LOAD AUDIO
# ===============================
y, sr = librosa.load(VIDEO_PATH, sr=SR)
print(f"Loaded match {MATCH_NUM} | duration {fmt_time(len(y)/sr)}")

S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

freqs = librosa.fft_frequencies(sr=SR)
mask = (freqs >= WHISTLE_LOW) & (freqs <= WHISTLE_HIGH)
S_w = S_db[mask]

# ===============================
# ENERGY + STICKY HYSTERESIS
# ===============================
# --- BAND-LIMITED RMS ENERGY (NEW) ---
energy = np.sqrt(np.mean((10 ** (S_w / 20)) ** 2, axis=0))
energy_db = 20 * np.log10(energy + 1e-12)

HIGH_THR = np.percentile(energy_db, 70)
LOW_THR  = np.percentile(energy_db, 45)

hold_frames = int(HOLD_TIME_SEC / (HOP / SR))
active = []
in_event = False
hold = 0

# --- IN-EVENT WHISTLE RE-TRIGGER PARAMS ---
WHISTLE_SALIENCE_THR = -35.0   # tune ±5 if needed
SUB_EVENT_COOLDOWN   = int(0.4 / (HOP / SR))  # 400 ms
last_sub_event = -10**9
def whistle_salience(spec):
    peak = spec.max()
    if peak < -60:
        return -np.inf
    active_bins = np.sum(spec > (peak - 6))
    return peak - active_bins * 2.5

for i, e in enumerate(energy_db):

    spec = S_w[:, i]
    sal = whistle_salience(spec)

    if not in_event:
        if e > HIGH_THR:
            in_event = True
            hold = hold_frames
            active.append(i)

    else:
        # 🔹 NORMAL CONTINUATION
        if e > LOW_THR:
            hold = hold_frames
            active.append(i)

        # 🔹 IN-EVENT WHISTLE RE-TRIGGER (CRITICAL FIX)
        elif (
            sal > WHISTLE_SALIENCE_THR
            and (i - last_sub_event) > SUB_EVENT_COOLDOWN
        ):
            active.append(i)
            last_sub_event = i
            hold = hold_frames

            print(
                f"[IN-EVENT WHISTLE] {fmt_time(i * HOP / SR)} "
                f"sal={sal:.1f}"
            )

        # 🔹 NORMAL DECAY
        else:
            hold -= 1
            if hold > 0:
                active.append(i)
            else:
                in_event = False


# ===============================
# GROUP FRAMES
# ===============================
groups = []
curr = [active[0]] if active else []

for f in active[1:]:
    if f - curr[-1] <= MAX_GAP_FRAMES:
        curr.append(f)
    else:
        groups.append(curr)
        curr = [f]
groups.append(curr)

# ===============================
# DRY FILTER
# ===============================
detections = []

for g in groups:
    if len(g) < MIN_FRAMES:
        continue

    tonal_groups = split_by_tonality(g, S_w, MIN_FRAMES)

    group_candidates = []   # 🔴 STEP 2: collect per-group

    for tg in tonal_groups:
        best_score = -np.inf
        core_frame = None

        for frame in tg:
            spec = S_w[:, frame]
            peak = spec.max()

            if peak < -60:
                continue

            active_bins = np.sum(spec > (peak - 6))
            score = peak - active_bins * 2.0

            if score > best_score:
                best_score = score
                core_frame = frame

        if core_frame is None:
            continue

        half = MIN_FRAMES // 2
        s = max(tg[0], core_frame - half)
        e = min(tg[-1], core_frame + half)

        if (e - s + 1) < MIN_FRAMES:
            continue

        start_sec = s * HOP / SR
        end_sec   = e * HOP / SR

        print(
            f"[CORE] group {fmt_time(g[0]*HOP/SR)} → {fmt_time(g[-1]*HOP/SR)} | "
            f"sub {fmt_time(tg[0]*HOP/SR)} → {fmt_time(tg[-1]*HOP/SR)} | "
            f"core={fmt_time(core_frame*HOP/SR)} score={best_score:.1f}"
        )

        segment = S_w[:, s:e + 1]

        # --- DRY PHYSICS CHECK (SOFT) ---
        fe = np.percentile(segment, 75, axis=0)
        ridge = np.sum(fe > (fe.mean() + fe.std()))
        grad_f = np.abs(np.diff(segment, axis=0)).mean()

        print(f"[CHECK] {fmt_time(start_sec)} ridge={ridge} grad_f={grad_f:.2f}")



        # --- AUDIO SNIPPET ---
        start_s = max(0, int((start_sec - PAD_BEFORE) * SR))
        end_s   = min(len(y), int((end_sec + PAD_AFTER) * SR))
        snippet = y[start_s:end_s]

        # --- SPECTRAL CHECKS ---
        flat = librosa.feature.spectral_flatness(y=snippet).mean()
        cent = librosa.feature.spectral_centroid(y=snippet, sr=SR).mean()
        noisy = flat > MAX_FLATNESS or cent > MAX_CENTROID

        freqs_w = freqs[mask]
        peak_std = peak_freq_std(segment, freqs_w)
        bw_hz = band_width_hz(segment, freqs_w)

        group_candidates.append({
            "start": start_sec,
            "end": end_sec,
            "audio": snippet,
            "noisy": noisy,
            "peak_std": peak_std,
            "bandwidth_hz": bw_hz,
            "core_score": best_score,
            "grad_f": grad_f,
            "ridge": ridge,
        })

    # 🔥 STEP 2: PICK ONE WINNER PER GROUP
    if not group_candidates:
        continue

    best = max(group_candidates, key=lambda d: d["core_score"])

    print(
        f"[GROUP-WINNER] {fmt_time(best['start'])} → "
        f"{fmt_time(best['end'])} score={best['core_score']:.1f}"
    )

    # 🔹 STEP 3: SOFT PHYSICS CHECK (NO FILTERING)
    best["physics_weak"] = False

    if best["grad_f"] < MIN_GRAD_F or best["ridge"] < MIN_RIDGE:
        best["physics_weak"] = True
    detections.append(best)


# ===============================
# TWO-STAGE MODEL ROUTING
# ===============================
accepted = []
ambiguous = []
rejected = 0

for d in detections:
    feats, X = extract_features(d["audio"])
    p1 = clf.predict_proba(X.reshape(1, -1))[0, 1]
    ts = f"{fmt_time(d['start'])} → {fmt_time(d['end'])}"

    # 1️⃣ STRONG MODEL
    if p1 >= CLS_THRESHOLD:
        accepted.append(d)
        print(f"[ACCEPT-1] {ts} p1={p1:.3f}")
        continue

    # 2️⃣ AMBIG RANGE → SECOND MODEL
    if AMBIG_MODEL_LOW <= p1 < CLS_THRESHOLD:
        p2 = ambig_clf.predict_proba(X.reshape(1, -1))[0, 1]

        if (
            p2 >= 0.60
            and d["peak_std"] < PEAK_STD_MAX
            and d["bandwidth_hz"] < BANDWIDTH_MAX
        ):
            accepted.append(d)
            print(f"[ACCEPT-2] {ts} p1={p1:.3f} p2={p2:.3f}")
        else:
            ambiguous.append(d)
            print(f"[AMBIG-2] {ts} p1={p1:.3f} p2={p2:.3f}")
        continue

    # 3️⃣ LOW CONFIDENCE FALLBACKS

    # 🔹 Strong tonal core → keep for HITL
    if d["core_score"] >= SCORE_OK:
        ambiguous.append(d)
        print(f"[AMBIG-CORE] {ts} p1={p1:.3f} core={d['core_score']:.1f}")
        continue

    # ❌ Ultra-low confidence → reject outright
    if p1 < 0.001:
        rejected += 1
        print(f"[REJECT-LOW] {ts} p1={p1:.3f}")
        continue

    # 🔻 Physics only matters if model is unsure
    if (
            PHYSICS_MIN_PROBA <= p1 < CLS_THRESHOLD
            and (d["noisy"] or physics_suspect(feats))
    ):
        ambiguous.append(d)
        print(f"[AMBIG-PHYS] {ts} p1={p1:.3f}")
        continue

    # 🔹 Truly nothing going for it
    rejected += 1
    print(f"[REJECT] {ts} p1={p1:.3f}")

print(f"\nSummary: accepted={len(accepted)} ambiguous={len(ambiguous)} rejected={rejected}")




# ===============================
# TEMPORAL MERGE (FINAL FIX)
# ===============================
MERGE_WINDOW_SEC = 1.0  # ← this is the key knob

all_events = []

for d in accepted:
    d["level"] = "accept"
    all_events.append(d)

for d in ambiguous:
    d["level"] = "ambig"
    all_events.append(d)

# sort by time
all_events.sort(key=lambda x: x["start"])

merged = []

for d in all_events:
    if not merged:
        merged.append(d)
        continue

    last = merged[-1]

    # If close in time → same whistle
    if d["start"] - last["end"] <= MERGE_WINDOW_SEC:
        # merge windows
        last["start"] = min(last["start"], d["start"])
        last["end"]   = max(last["end"], d["end"])

        # escalate confidence
        if d["level"] == "accept":
            last["level"] = "accept"

        print(
            f"[MERGE] {fmt_time(d['start'])} merged into "
            f"{fmt_time(last['start'])} → {fmt_time(last['end'])}"
        )
    else:
        merged.append(d)

# rebuild accepted / ambiguous
accepted = [d for d in merged if d["level"] == "accept"]
ambiguous = [d for d in merged if d["level"] != "accept"]

print(
    f"\n[MERGE SUMMARY] accepted={len(accepted)} "
    f"ambiguous={len(ambiguous)}"
)
# ===============================
# SAVE ACCEPTED + HITL PIPELINE
# ===============================

AMBIG_DIR = Path(fr"{BASE_OUTPUT_DIR}\ambiguous\match{MATCH_NUM}")
TRAINING_CSV = Path(fr"{BASE_OUTPUT_DIR}\training_with_uid.csv")

def append_training_row(uid, X, label):
    header = [
        "uid",
        "rms","flatness","centroid","band_energy","peak_freq",
        "ridge","grad_t","grad_f",
        "rolloff","bandwidth","zcr","contrast","tonnetz",
        "mfcc1","mfcc2","mfcc3","mfcc4","mfcc5",
        "label"
    ]

    file_exists = TRAINING_CSV.exists()
    if file_exists:
        existing = set(pd.read_csv(TRAINING_CSV)["uid"])
        if uid in existing:
            return

    with open(TRAINING_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([uid, *X.tolist(), label])


# ---------- SAVE AUTO-ACCEPTED ----------
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

accepted_clips = []
for d in accepted:
    c = save_clip(d["start"], d["end"], OUTPUT_DIR)
    accepted_clips.append(c)


# ---------- SAVE AMBIGUOUS ----------
if AMBIG_DIR.exists():
    shutil.rmtree(AMBIG_DIR)
AMBIG_DIR.mkdir(parents=True)

ambig_items = []
for d in ambiguous:
    c = save_clip(d["start"], d["end"], AMBIG_DIR)
    ambig_items.append((c, d))


# ===============================
# HUMAN IN THE LOOP
# ===============================
print("\n=== HUMAN IN THE LOOP ===")
print("[1] whistle | [2] noise | [q] quit")

manual_accept = []
count = 0
for path, d in ambig_items:
    count+=1
    ts = f"{fmt_time(d['start'])} → {fmt_time(d['end'])}"
    print(f"\n{count}. Reviewing: {path.name} | {ts}")
    os.startfile(path)

    while True:
        ans = input("Decision [yes = 1/ no = 2/ q]: ").strip().lower()
        if ans in ("1", "2", "q"):
            break

    if ans == "q":
        break

    feats, X = extract_features(d["audio"])
    uid = f"match{MATCH_NUM}_{int(d['start']*1000):010d}_{int(d['end']*1000):010d}"

    if ans == "1":
        manual_accept.append(d)
        append_training_row(uid, X, 1)
        print("✔ labeled WHISTLE")
    else:
        append_training_row(uid, X, 0)
        print("✘ labeled NOISE")


# ===============================
# FINAL MERGE + CONCAT
# ===============================
final = accepted + manual_accept
final.sort(key=lambda x: x["start"])

clips = [save_clip(d["start"], d["end"], OUTPUT_DIR) for d in final]

with open(OUTPUT_DIR / "concat.txt", "w") as f:
    for c in clips:
        f.write(f"file '{c.resolve().as_posix()}'\n")

subprocess.run([
    FFMPEG, "-y",
    "-f", "concat", "-safe", "0",
    "-i", str(OUTPUT_DIR / "concat.txt"),
    "-c", "copy",
    FINAL_OUTPUT
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("\nDONE:", FINAL_OUTPUT)