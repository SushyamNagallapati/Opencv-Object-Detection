import numpy as np
import cv2

capture = cv2.VideoCapture("Videos/drive.mp4")

parameter_shitomasi = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7)
#100 denotes the no. of corners we want to detect, the higher the value, the more objects the algorith will track.
#The quality level (0.3) helps filter out weak corners. Only corners with a score above the threshold (450 in this case) are kept. This ensures the algorithm uses only the best corners for better results.
#If a corner is detected on the image, and the next corner must be detected 7 pixels to left, right, up or down. It is the distance of the closest edge

#KLT method, which uses the concept of pyramids
parameters_lucas_kanade = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#The winSize indicates the size of the pyramids ie, when we multiply 15*15 = 225pixels is the minimum size of the pyramid.
#There can be many levels(1, 2, 3, 4, 5...) in the pyramid
#Both parameters will indicate when the algorithm will finish the execution. If you set a higher number for cv2.TERM_CRITERIA_EPS,
                                                                             #the algorithm will take a longer time to execute, but the results
                                                                             #may be better. Regarding the second parameter, it will indicate
                                                                             #how sensible the algorithm is related to the changes of the object.

colors = np.random.randint(0, 255, (100, 3)) #As we are going to detect 100 corners in the first frame of the video, we are going to generate 100 random numbers
                                             #And when we are working with the RGB color space that are three channels, one channel for R, one channel
                                             #for G and one channel for B, for this reason, we need to put number three.

# print(np.shape(colors))
# print(colors)

ok, frame = capture.read()
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Init Frame", frame_gray_init)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# capture.release()

edges = cv2.goodFeaturesToTrack(frame_gray_init, mask= None, **parameter_shitomasi) #This is the intialization of the algorithm.
                                                                                    #The Harris Corner detector is being used.

# print(len(edges))
# print(edges)

mask = np.zeros_like(frame)
print(np.shape(mask)) #this gives the dimensions of the first frame of the video, the width, the height
                      #and the number of channels.
print(mask) #we have a matrix composed of only zeros when we are working with the RGB format.
            #Value zero is related to the black color, while value 255 is related to the yellow color.

        
