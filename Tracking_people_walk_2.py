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
        detections = cascade.detectMultiScale(frame_gray, minSize=(60, 60))  # "minsize" denotes for the minimum size the object to be detected (so it does not detect the rock)
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("Detection", frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if x > 0:
                print("Haarscade Detection")
                return x, y, w, h

# boundingBox = detect()
boundingBox = cv2.selectROI(frame)
# print(boundingBox)

ok = tracker.init(frame, boundingBox)
colors = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    ok, frame = video.read()
    if not ok:
        break

    ok, boundingBox = tracker.update(frame)
    if ok:
        (x, y, w, h) = [int(v) for v in boundingBox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors)
    else:
        print("Tracking Failure! We will execute the Haarcascade detector")
        boundingBox = detect()
        tracker = cv2.legacy.TrackerCSRT_create()
        tracker.init(frame, boundingBox)

    cv2.imshow("people Tracking", frame)
    k = cv2.waitKey(1) & 0XFF
    if k == 27: #Esc key
        break

