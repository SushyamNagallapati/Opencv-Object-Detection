import cv2
import sys
from random import randint

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = tracker_types[6]
# print(tracker_type)

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy.TrackerCSRT_create()

# print(tracker)

video = cv2.VideoCapture('Videos/race.mp4')
if not video.isOpened():
    print("Error while loading the video!")
    sys.exit()

ok, frame = video.read()
if not ok:
    print("Error while loading the frame!")
    sys.exit()

# print(ok)

boundingBox = cv2.selectROI(frame) #Region of Interest

print(boundingBox)

ok = tracker.init(frame, boundingBox)
print(ok)

colors = (randint(0, 255), randint(0, 255), randint(0, 255))  #RGB
print(colors)


while True:
    ok, frame = video.read()
    # print(ok)
    if not ok:
        break

    ok, boundingBox = tracker.update(frame)
    # print(ok, boundingBox)

    if ok == True:
        (x, y, w, h) = [int(v) for v in boundingBox]
        # print(x, y, w, h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:
        cv2.putText(frame, "tracking failure!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.putText(frame, tracker_type, (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0XFF == 27: #esc key
        break

  
