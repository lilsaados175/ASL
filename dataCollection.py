import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import sys

# 1. Open the webcam and check
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: could not open webcam.")
    sys.exit(1)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Data/A"
counter = 0

while True:
    # 2. Grab a frame and check
    success, img = cap.read()
    if not success or img is None:
        print("Warning: empty frame.")
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # 3. Clamp crop coordinates to image bounds
        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, img.shape[1])
        y2 = min(y + h + offset, img.shape[0])

        imgCrop = img[y1:y2, x1:x2]

        # 4. Skip if the crop is empty
        if imgCrop.size == 0:
            print("Skipped an empty crop.")
        else:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = (y2 - y1) / (x2 - x1)

            if aspectRatio > 1:
                k = imgSize / (y2 - y1)
                wCal = math.ceil(k * (x2 - x1))
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / (x2 - x1)
                hCal = math.ceil(k * (y2 - y1))
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # 5. Show results
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s") and 'imgWhite' in locals():
        counter += 1
        filename = f'{folder}/Image_{time.time():.0f}.jpg'
        cv2.imwrite(filename, imgWhite)
        print(f"Saved {filename}  ({counter})")
    elif key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()