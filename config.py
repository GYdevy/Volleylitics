import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
VIDEO_DIR = BASE_DIR / "videos"
RALLY_BASE = BASE_DIR / "rally_segmentator"

# =====================
# MODEL CONFIG
# =====================

MODELS_DIR = BASE_DIR / "models"

WHISTLE_MODEL = Path(
    os.environ.get(
        "VOLLEYLITICS_WHISTLE_MODEL",
        MODELS_DIR / "best_resnet_whistle_mix.pth",
    )
)
RALLY_MODEL = Path(
    os.environ.get(
        "VOLLEYLITICS_RALLY_MODEL",
        MODELS_DIR / "rally_model_best.pth",
    )
)
BALL_MODEL = Path(
    os.environ.get(
        "VOLLEYLITICS_BALL_MODEL",
        MODELS_DIR / "temp_best_ball.pt",
    )
)


# =========================
# DATA SPLIT CONFIG
# =========================

SPLIT = {
    "train": [
        "match1","match2","match3","match8","match9",
        "match10","match11","match14","match15","match16"
    ],

    "val": [
        "match13",   # FHD
        "match7",    # HD
    ],

    "test": [
        "match17",   # FHD
        "match18",   # FHD
        "match4",    # HD
    ]
}


# =====================
# DOMAIN GROUPS
# =====================

FHD_MATCHES = {
    "match1","match8","match9","match10","match11",
    "match13","match14","match15","match16","match17","match18"
}

HD_MATCHES = {
    "match2","match3","match4","match7"
}



