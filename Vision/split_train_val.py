import random, shutil
from pathlib import Path

train_img_dir = Path(r"/dataset/images/train")
train_lbl_dir = Path(r"/dataset/labels/train")
val_img_dir = Path(r"/dataset/images/val")
val_lbl_dir = Path(r"/dataset/labels/val")

val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

# move 15% to val
imgs = list(train_img_dir.glob("*.jpg"))
random.shuffle(imgs)
val_count = int(len(imgs) * 0.15)

for img in imgs[:val_count]:
    lbl = train_lbl_dir / (img.stem + ".txt")
    shutil.move(str(img), val_img_dir / img.name)
    shutil.move(str(lbl), val_lbl_dir / lbl.name)

print(f"✅ Moved {val_count} images to validation set.")
