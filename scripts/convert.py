from pathlib import Path
import shutil

BASE = Path("dataset_yolo")

for split in ["train", "val"]:
    img_src = BASE / "images" / split
    lbl_src = BASE / "labels" / split

    img_dst = BASE / "images_clean" / split
    lbl_dst = BASE / "labels_clean" / split

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    count = 0

    for label in lbl_src.glob("*.txt"):
        img = img_src / (label.stem + ".jpg")

        if img.exists():
            shutil.copy2(img, img_dst / img.name)
            shutil.copy2(label, lbl_dst / label.name)
            count += 1

    print(f"{split}: copied {count}")