import cv2
import json
import numpy as np
import argparse

POINT_LABELS = [
    "Click BOTTOM-LEFT court corner",
    "Click BOTTOM-RIGHT court corner",
    "Click TOP-RIGHT lower-half corner (near net)",
    "Click TOP-LEFT lower-half corner (near net)",
    "Click LEFT edge of the net",
    "Click RIGHT edge of the net",
]

points = []
frame = None
base_frame = None


def redraw():
    global frame
    frame = base_frame.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if len(points) < len(POINT_LABELS):
        msg = POINT_LABELS[len(points)]
    else:
        msg = "Done. ESC to finish, BACKSPACE to undo."

    cv2.putText(
        frame,
        msg,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(POINT_LABELS):
        points.append((x, y))
        print(f"{len(points)}. {POINT_LABELS[len(points)-1]} -> ({x}, {y})")
        redraw()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to source match video")
    parser.add_argument("--out", required=True, help="Path to output calibration json")
    parser.add_argument("--seek-minutes", type=float, default=10.0, help="Seek position in minutes")
    args = parser.parse_args()

    global base_frame
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.seek_minutes * 60 * 1000)
    ret, base_frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read frame for calibration")

    redraw()

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", click)

    while True:
        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key in (8, 127):  # backspace/delete
            if points:
                removed = points.pop()
                print(f"Removed point {removed}")
                redraw()

    cv2.destroyAllWindows()

    if len(points) != 6:
        raise RuntimeError(f"Expected 6 clicks, got {len(points)}")

    img_pts = np.array(points[:4], dtype=np.float32).tolist()
    net_left = points[4]
    net_right = points[5]
    net_y = int((net_left[1] + net_right[1]) / 2)

    data = {
        "img_pts": img_pts,
        "net_left_x": int(net_left[0]),
        "net_right_x": int(net_right[0]),
        "net_y_img": net_y,
    }

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved calibration to {args.out}")


if __name__ == "__main__":
    main()
