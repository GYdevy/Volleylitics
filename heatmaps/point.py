import cv2

VIDEO = "/mnt/hdd/videos/match17.mp4"

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"{x}, {y}")
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0,0,255), -1)

cap = cv2.VideoCapture(VIDEO)

#skip 10 minutes
cap.set(cv2.CAP_PROP_POS_MSEC, 10 * 60 * 1000)

ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame at 10 minutes")
    exit()

cv2.imshow("Click court corners", frame)
cv2.setMouseCallback("Click court corners", click)

while True:
    cv2.imshow("Click court corners", frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

print("Points:", points)
