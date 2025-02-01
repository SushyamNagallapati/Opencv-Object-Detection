import cv2
import numpy as np

capture = cv2.VideoCapture("Videos/walking.avi")

ok, first_frame = capture.read()
frame_gray_init = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(first_frame)

# print(np.shape(first_frame))
# print(np.shape(hsv))
# print(first_frame)
# print(hsv)

hsv[...,1] = 255 #the ... is used cause when we run the hsv alone we get 3 sq brackets. In those 3 sq brackets we are selecting the 1st position ie, in the order 0, 1, 2.

print(hsv)

while True:
    ok, frame = capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #refer in net about calcOpticalFlowFarneback opencv
    flow = cv2.calcOpticalFlowFarneback(frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)

    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1]) #both values represent the x and y
                                        #cv2.cartToPolar() is an Opencv function to get both the values(magnitude and angle) to know the direction of the arrows.
                                        #to predict where the object is going or the direction, and we will send here as parameter flow

    hsv[...,0] = angle * (180 / (np.pi / 2)) #this is an algebra equation to find the direction of the arrow
    hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) #in position 2 we are normalizing the magnitudes

    frame_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) #We are converting from HSV to BGR, because we have color space in hsv matrix which is used
                                                     #to perform the calculations regarding the optical flow algorithm

    cv2.imshow("Dense Optical Flow", frame_rgb)
    if cv2.waitKey(1) == 13: #Enter key
        break

    frame_gray_init = frame_gray #We always need to compare the previous frame to next frame

cv2.destroyAllWindows()
capture.release()





