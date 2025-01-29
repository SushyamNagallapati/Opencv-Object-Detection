import cv2
import numpy as np

capture = cv2.VideoCapture(0)

ok, frame = capture.read()

frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

parameters_lucas_kanade = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def select_point(event, x, y, flags, params): #select_point is an opencv function, it holds the parameters: event, x, y, flags and params
    global point, selected_point, old_points

    if event == cv2.EVENT_LBUTTONDOWN:  #We are clicking the left button of the mouse
        point = (x, y) #When we click in a specific position of the window, the x and y coordinates(from select_point)
                       #will be copied to this variable here (x, y)
        selected_point = True
        old_points = np.array([[x, y]], dtype=np.float32) #we are keeping the current points and the previous points

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point) #we are associating this function to the window called frame.
                                                        #Every time we create a window with this name, it will be possible
                                                        #to execute this function to get the points according to the left button down events
                                                        #on the mouse

selected_point = False #This means that no point is selected

point = ()
old_points = np.array([[]])

mask = np.zeros_like(frame)

while True:
    ok, frame = capture.read()

    if not ok: #If we are not able to read the frame, the video will end
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if selected_point is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2) #We draw a circle on the point

        #We get the new points(new_points) this code will update the location of the points. We send the Lucas kanade Parameters
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, old_points, None, **parameters_lucas_kanade)


        frame_gray_init = frame_gray.copy() #We make a copy of the frame
        old_points = new_points #Store the old points

        x, y = new_points.ravel() #We get the current location
        j, k = old_points.ravel() #We get the next location

        mask = cv2.line(mask, (int(x), int(y)), (int(j), int(k)), (0, 255, 255), 2)
        frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    img = cv2.add(frame, mask) #we are adding the frame and mask
    cv2.imshow("Frame", img)
    cv2.imshow("Frame 2", mask)

    key = cv2.waitKey(1)
    if key == 27: #Esc key
        break

cv2.destroyAllWindows()
capture.release()


