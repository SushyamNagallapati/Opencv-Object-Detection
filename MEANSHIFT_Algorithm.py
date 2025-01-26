import cv2
import time
from imutils.video import VideoStream

capture = VideoStream(src=0).start() #src = 0 represents the default laptop webcam
time.sleep(1.0) #we are going to wait for 1 sec before capturing the images from webcam. It allows the camera to adapt to the lighting conditions

capture = cv2.VideoCapture(0)
ok, frame = capture.read()

print(ok)

boundingBox = cv2.selectROI(frame)
x, y, w, h = boundingBox
tracking_window = (x, y, w, h)
print(tracking_window)

region_of_interest = frame[y:y+h, x:x+w]

cv2.imshow("ROI", region_of_interest)
cv2.waitKey(0)
