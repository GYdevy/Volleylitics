from pathlib import Path


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
TIMELINE_CSV = (
    Path(BASE_OUTPUT_DIR)
    / "timelines"
    / f"match{MATCH_NUM}_whistles.csv"
)
TIMELINE_CSV.parent.mkdir(exist_ok=True)

# AUDIO
SR = 22050
N_FFT = 2048
HOP = 128

WHISTLE_LOW = 3700
WHISTLE_HIGH = 4200

PAD_BEFORE = 0.10
PAD_AFTER  = 0.10

MIN_DURATION_SEC = 0.06
MIN_FRAMES = int(MIN_DURATION_SEC / (HOP / SR))

# THRESHOLDS
MAX_GAP_FRAMES = 4
HOLD_TIME_SEC = 0.2

MIN_RIDGE  = 11
MIN_GRAD_F = 4.7

MAX_FLATNESS = 0.025
MAX_CENTROID = 3000
BANDWIDTH_MAX = 180.0
WHISTLE_SALIENCE_THR = -35.0
SUB_EVENT_COOLDOWN = 0.4
SCORE_OK = -24.0
CLS_THRESHOLD = 0.72
AMBIG_MODEL_LOW = 0.10
PHYSICS_MIN_PROBA = 0.02
PEAK_STD_MAX = 120.0
