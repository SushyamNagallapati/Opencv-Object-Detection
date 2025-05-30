import cv2
import sys
from random import randint

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

def tracker_by_name(tracker_type):
    if tracker_type == tracker_types[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == tracker_types[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == tracker_types[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == tracker_types[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == tracker_types[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == tracker_types[5]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == tracker_types[6]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print("Invalid Input! The Available Trackers are: ")
        for t in tracker_types:
            print(t)
    return tracker

# print(tracker_by_name("CSRT"))

video = cv2.VideoCapture("Videos/race.mp4")
if not video.isOpened():
    print("Error while loading the Video!")
    sys.exit()

ok, frame = video.read()

boundingBoxes = []
colors = []

while True:
    boundingBox = cv2.selectROI("MultiTracker", frame)
    boundingBoxes.append(boundingBox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press Q to start and quit Tracking")
    print("Press any other key to select the next Object")
    k = cv2.waitKey(0) & 0XFF
    if k == 113: #Q - Quit
        break

print(boundingBoxes)
print(colors)


