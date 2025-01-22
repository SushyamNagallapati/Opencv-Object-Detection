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
