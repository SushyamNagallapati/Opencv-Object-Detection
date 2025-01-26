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

#We know the range is from 0 to 255, so we need to convert them from 0 to 1
roi_histogram = cv2.normalize(roi_histogram, roi_histogram, 0, 255, cv2.NORM_MINMAX)

parameters = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ok, frame = capture.read()
    if ok == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_histogram, [0, 180], 1) #We are comparing the "hsv" to the "roi_histogram" ie, the object that we want to track. Check about the calcBackProject() function on Web.
        ok, tracking_window = cv2.meanShift(dst, (x, y, w, h), parameters)
        x, y, w, h = tracking_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("MEANSHIFT Tracking", frame)

        cv2.imshow("dst", dst) #to see the density where the tracker focuses

        if cv2.waitKey(1) == 13: #Esc key
            break

    else:
        break

cv2.destroyAllWindows()
capture.release() #Used to release the memory

