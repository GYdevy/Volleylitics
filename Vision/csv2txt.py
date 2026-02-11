import os
import csv
import shutil
from pathlib import Path

# === CONFIG ===
CSV_PATH = Path(r"/data/labels.csv")
TRAIN_DIR = Path(r"/data/matches")
VAL_DIR = Path(r"/data/val")
OUTPUT_DIR = Path(r"/dataset")

BALL_RADIUS = 0.025  # approximate normalized ball radius for YOLO box size
# ===============


def ensure_dirs():
    """Create YOLO-style dataset folder structure."""
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def copy_and_label(src_img: Path, split: str, x=None, y=None, has_ball=False):
    """Copy image and create YOLO label file."""
    dst_img = OUTPUT_DIR / "images" / split / src_img.name
    dst_lbl = OUTPUT_DIR / "labels" / split / (src_img.stem + ".txt")

    # Copy image
    if not dst_img.exists():
        shutil.copy2(src_img, dst_img)

    # Write label
    with open(dst_lbl, "w", encoding="utf-8") as f:
        if has_ball and x is not None and y is not None:
            w = BALL_RADIUS * 2
            h = BALL_RADIUS * 2
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    ensure_dirs()
    total_train, total_val, missing = 0, 0, 0

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: (r["match"], r["frame"]))

        for row in rows:
            match = row["match"].strip()
            frame = row["frame"].strip()

            has_ball = row["has_ball"].strip().lower() == "true"
            x = float(row["x"]) if row["x"] else None
            y = float(row["y"]) if row["y"] else None

            # Check whether frame belongs to validation
            val_img = VAL_DIR / frame
            if val_img.exists():
                src_img = val_img
                split = "val"
                total_val += 1
            else:
                src_img = TRAIN_DIR / match / frame
                split = "train"
                total_train += 1

            if not src_img.exists():
                print(f"⚠️ Missing image: {src_img}")
                missing += 1
                continue

            copy_and_label(src_img, split, x, y, has_ball)

    # Write YOLO data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {OUTPUT_DIR.as_posix()}/images/train\n")
        f.write(f"val: {OUTPUT_DIR.as_posix()}/images/val\n\n")
        f.write("nc: 1\n")
        f.write("names: ['ball']\n")

    print("\n✅ YOLO dataset successfully built!")
    print(f"📁 Output folder: {OUTPUT_DIR}")
    print(f"📄 data.yaml written at: {yaml_path}")
    print(f"📊 Train images: {total_train}, Val images: {total_val}, Missing: {missing}")


if __name__ == "__main__":
    main()
