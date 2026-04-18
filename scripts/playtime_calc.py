import json

RALLIES_FILE = "match17_rallies_with_hitl.json"
GT_FILE = "match17_rallies_from_whistles.json"  

VIDEO_DURATION = 4148  


def total_duration(rallies):
    return sum(r["end"] - r["start"] for r in rallies)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    detected = load_json(RALLIES_FILE)
    detected_time = total_duration(detected)

    print("===== DETECTED =====")
    print(f"Rallies: {len(detected)}")
    print(f"Total play time: {detected_time:.2f} sec ({detected_time/60:.2f} min)")

    
    try:
        gt = load_json(GT_FILE)
        gt_time = total_duration(gt)

        print("\n===== GROUND TRUTH =====")
        print(f"Rallies: {len(gt)}")
        print(f"Total play time: {gt_time:.2f} sec ({gt_time/60:.2f} min)")

        diff = detected_time - gt_time
        print("\n===== DIFF =====")
        print(f"Difference: {diff:.2f} sec ({diff/60:.2f} min)")

    except FileNotFoundError:
        print("\n(No GT file found, skipping comparison)")

    
    if VIDEO_DURATION:
        pct = (detected_time / VIDEO_DURATION) * 100
        print("\n===== VIDEO COVERAGE =====")
        print(f"Playtime %: {pct:.2f}%")


if __name__ == "__main__":
    main()
