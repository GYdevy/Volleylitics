import os
from pathlib import Path
from PIL import Image, ImageDraw

# === CONFIG ===
IMG_DIR = Path(r"/dataset/images/val")
LBL_DIR = Path(r"/dataset/labels/val")
OUT_DIR = Path(r"/Vision/dataset\sanity_check_val")
OUT_DIR.mkdir(exist_ok=True)
# ===============

def draw_yolo_box(img_path, lbl_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)

            # Convert normalized YOLO coords -> pixel coordinates
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.ellipse([xc * w - 3, yc * h - 3, xc * w + 3, yc * h + 3], fill="red")

    return img

def main():
    imgs = list(IMG_DIR.glob("*.jpg"))
    if not imgs:
        print("⚠️ No images found in val folder.")
        return

    print(f"🧪 Checking {len(imgs)} validation images...")
    for img_path in imgs:
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"⚠️ Missing label for {img_path.name}")
            continue

        img_out = draw_yolo_box(img_path, lbl_path)
        out_path = OUT_DIR / img_path.name
        img_out.save(out_path)

    print(f"\n✅ Done! Output images saved to:\n{OUT_DIR}")

if __name__ == "__main__":
    main()
