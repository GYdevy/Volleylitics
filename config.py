from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
WHISTLE_MODEL = "best_resnet_whistle_mix.pth"
RALLY_MODEL = "rally_model_best.pth"

VIDEO_DIR = "/mnt/hdd/videos"

MODEL_DIR = BASE_DIR / "rally_segmentator" / "models_dir"

MATCH_ID = "match17"

RALLY_BASE = BASE_DIR / "rally_segmentator"

RALLIES_DIR = RALLY_BASE / "output" / MATCH_ID 



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



