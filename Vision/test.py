import re
from pathlib import Path

# === CONFIG ===
TARGET_FOLDERS = [
    r"D:\Volleyballey\data\matches\match5",
    r"D:\Volleyballey\dataset\images\train",
    r"D:\Volleyballey\dataset\labels\train",
    r"D:\Volleyballey\dataset\images\val",
    r"D:\Volleyballey\dataset\labels\val",
]
# ===============

# Pattern: keep only the last "match<number>_frame_<digits>.<ext>"
pattern = re.compile(r"(match\d+_frame_\d+\.(?:jpg|txt))$", re.IGNORECASE)

def clean_prefixes(folder: Path):
    folder = Path(folder)
    if not folder.exists():
        return

    for file in sorted(folder.glob("*.*")):
        m = pattern.search(file.name)
        if not m:
            continue

        clean_name = m.group(1)
        if clean_name != file.name:
            new_path = file.parent / clean_name
            if new_path.exists():
                print(f"⚠️ Skipping (already exists): {new_path.name}")
                continue

            file.rename(new_path)
            print(f"✅ {file.name} → {clean_name}")

def main():
    print("🔧 Cleaning redundant prefixes (e.g., match5_match2_ → match2_...)")
    for folder in TARGET_FOLDERS:
        print(f"\n📁 {folder}")
        clean_prefixes(folder)
    print("\n🎯 Done! All redundant prefixes cleaned up.")

if __name__ == "__main__":
    main()
