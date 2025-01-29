import numpy as np
import cv2

capture = cv2.VideoCapture("Videos/walking.avi")

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

print(len(edges))
print(edges)

mask = np.zeros_like(frame) #the zeros_like related to zero, which represents only the black color
# print(np.shape(mask)) #this gives the dimensions of the first frame of the video, the width, the height
                      #and the number of channels.
# print(mask) #we have a matrix composed of only zeros when we are working with the RGB format.
            #Value zero is related to the black color, while value 255 is related to the yellow color.

while True:
    ok, frame = capture.read()

    if not ok: #If we are not able to read the frame, the video will end
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, edges, None, **parameters_lucas_kanade) #Calling the Optical Flow method
                                                                                      #We are calling the current frame(frame_gray) and initial frame(frame_gray_init)

    news = new_edges[status == 1] #The new variable which will be equal to new edges, the new X and y position of each one of the corners.
                                  #When the status is equal to one, we can keep tracking the edges.

    olds = edges[status == 1] #old refers to the previous position

    for i, (new, old) in enumerate(zip(news, olds)):
        a, b = new.ravel() #The a and b variables will store the new value or  the new position of the corner
        c, d = old.ravel() #The c and d variables will store the previous position of the corner

        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 2) #Let's create another variable mask Cv2 line mask. Let's send here A and B the new position,
                                                                        #C and D the previous position. Let's access colors in position i When we use enumerate function we are going
                                                                        #to copy to the i variable the current execution of the for loop. For example, in the first time we
                                                                        #run this for loop variable, i will be equal to zero. Then it will be equal to one, two, three and so on.
                                                                        #And we are going to use this variable to access this list of colors.

        frame = cv2.circle(frame, (int(a), int(b)), 5, colors[i].tolist(), -1)

        img = cv2.add(frame, mask)
        cv2.imshow("Optical Flow Sparse", img) #We can also put "frame"(only the pints will be visible) and "mask"(which draws the lines in a black bg)

        if cv2.waitKey(1) == 13: #Enter
            break

        frame_gray_init = frame_gray.copy() #This is a very important code because at the first execution we are using the first frame of the video to initialize the algorithm.
                                            #However, when we keep processing the video, we need to compare the previous frame with the next frame.
                                            #So we need to keep updating this variable

        edges = news.reshape(-1, 1, 2) #So now we can just put here news and in order to have in the correct format, let's type reshape
                                       # -1, 1 and 2, just to put in the correct format to send to the optical flow function here

cv2.destroyAllWindows()
capture.release()

