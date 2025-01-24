import cv2, sys, os
from random import randint

if not (os.path.isfile("goturn.caffemodel") and os.path.isfile("goturn.prototxt")):
    print("Files not found!")
    sys.exit()

tracker = cv2.TrackerGOTURN_create() #For the GO TURN algorithm we should not use 'legacy' package

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
# print(boundingBox)

ok = tracker.init(frame, boundingBox)
# print(ok)

colors = (randint(0, 255), randint(0, 255), randint(0, 255))  #RGB
# print(colors)

while True:
    ok, frame = video.read()
    # print(ok)
    if not ok:
        break

    ok, boundingBox = tracker.update(frame)
    # print(ok, boundingBox)

    if ok == True:
        (x, y, w, h) = [int(v) for v in boundingBox] #v is the known as the variable ie; (x, y, w, h), used in bounding box
        # print(x, y, w, h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 10)
    else:
        cv2.putText(frame, "tracking failure!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.putText(frame, "GOTURN", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0XFF == 27: #esc key
        break
