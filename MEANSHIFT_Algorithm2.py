import cv2
import time
from imutils.video import VideoStream

capture = VideoStream(src=0).start() #src = 0 represents the default laptop webcam
time.sleep(1.0) #we are going to wait for 1 sec before capturing the images from webcam. It allows the camera to adapt to the lighting conditions

capture = cv2.VideoCapture(0)
ok, frame = capture.read()

# print(ok)

boundingBox = cv2.selectROI(frame)
x, y, w, h = boundingBox
tracking_window = (x, y, w, h)
print(tracking_window)

roi = frame[y:y+h, x:x+w] #We need to convert the image from RGB --> BGR using HSV format, this is how MEANSHIFT algorithm is created.
# cv2.imshow("ROI", roi)
# cv2.waitKey(0)

#HSV
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# cv2.imshow("HSV_ROI", hsv_region_of_interest)
# cv2.waitKey(0)

#Refer about Histogram in Web "Histogram Calculation in OpenCV"
roi_histogram = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

import matplotlib.pyplot as plt
plt.hist(roi.ravel(), 180, [0, 180]) #we need to convert the ROI from Matrix to Vector format so ravel() is used.
# plt.show()       #Uncomment this line to see the Histogram graph for the selected part of the image
# cv2.waitKey(0)   #""

#We know the range is from 0 - 255, so we need to convert them from 0 to 1
roi_histogram = cv2.normalize(roi_histogram, roi_histogram, 0, 255, cv2.NORM_MINMAX)

