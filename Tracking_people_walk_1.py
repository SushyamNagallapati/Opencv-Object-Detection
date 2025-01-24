import cv2, sys
from random import randint


tracker = cv2.legacy.TrackerCSRT_create()

video = cv2.VideoCapture('Videos/walking.avi')
if not video.isOpened():
    print("Error while loading the video!")
    sys.exit()

ok, frame = video.read()
if not ok:
    print("Error while loading the frame!")
    sys.exit(0)

cascade = cv2.CascadeClassifier("cascade/fullbody.xml")

def detect():
    while True:
        ok, frame = video.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(frame_gray)
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("Detection", frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if x > 0:
                print("Haarscade Detection")
                return x, y, w, h

boundingBox = detect()
print(boundingBox)
