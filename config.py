from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
WHISTLE_MODEL = "best_resnet_whistle_mix.pth"
RALLY_MODEL = "rally_model_best.pth"

VIDEO_DIR = "/mnt/hdd/videos"

MODEL_DIR = BASE_DIR / "rally_segmentator" / "models_dir"

MATCH_ID = "match18"

RALLY_BASE = BASE_DIR / "rally_segmentator"

RALLIES_DIR = RALLY_BASE / "output" / MATCH_ID 
SEGMENTS = [
    (4*60 + 40, 24*60 + 2),
    (26*60 + 25, 50*60 + 32),
    (53*60 + 18, 1*3600 + 13*60 + 19)
]


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



