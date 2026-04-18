import os
import shutil
import soundfile as sf

# ==============================
# CONFIG
# ==============================

DATASET_DIR = r"E:\Volleyballey\cnn_dataset_by_match"
AUDIT_DIR = os.path.join(DATASET_DIR, "audit")

HIGH_NEG_DIR = os.path.join(AUDIT_DIR, "high_prob_neg")
LOW_POS_DIR = os.path.join(AUDIT_DIR, "low_prob_pos")

# ==============================
# AUDIO PLAYER
# ==============================

def play_audio(path):
    data, sr = sf.read(path)
    print(f"\nPlaying: {os.path.basename(path)}")
    sd.play(data, sr)
    sd.wait()

# ==============================
# MATCH EXTRACTION
# ==============================

def extract_match_id(filename):
    parts = filename.split("_")
    for p in parts:
        if p.startswith("match"):
            return p
    return None

# ==============================
# PROCESS FILE
# ==============================

def process_file(filepath):

    filename = os.path.basename(filepath)
    match_id = extract_match_id(filename)

    if match_id is None:
        print("Could not detect match ID:", filename)
        return

    play_audio(filepath)

    while True:
        answer = input("Is this a whistle? (y = yes, n = no, q = quit): ").strip().lower()

        if answer == "q":
            return "quit"

        if answer not in ["y", "n"]:
            print("Invalid input. Try again.")
            continue

        target_label = "pos" if answer == "y" else "neg"
        target_dir = os.path.join(DATASET_DIR, match_id, target_label)

        os.makedirs(target_dir, exist_ok=True)

        dst = os.path.join(target_dir, filename)
        shutil.move(filepath, dst)

        print(f"Moved to {match_id}/{target_label}")
        break

    return "ok"

# ==============================
# MAIN
# ==============================

def run_audit(folder):

    files = sorted(os.listdir(folder))
    files = [os.path.join(folder, f) for f in files if f.endswith(".wav")]

    print(f"\nFound {len(files)} files in {folder}")

    for f in files:
        result = process_file(f)
        if result == "quit":
            break

if __name__ == "__main__":

    print("1 = Review high_prob_neg")
    print("2 = Review low_prob_pos")

    choice = input("Choose folder: ").strip()

    if choice == "1":
        run_audit(HIGH_NEG_DIR)
    elif choice == "2":
        run_audit(LOW_POS_DIR)
    else:
        print("Invalid choice.")