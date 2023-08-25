#Create a program that can detect and track a specific color object in a video stream using OpenCV. 
#you'll need to define the specific HSV color range to track and create a binary mask using cv2.inRange. 
#Find contours using cv2.findContours and select the largest one by calculating its area. 
#The centroid of the contour can be calculated using moments, and 
#you can use functions like cv2.circle to draw a tracking indicator around the detected object. 


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    
    #lower and upper bound for the green color in the HSV color space
    lower_color_range = np.array([20,20,100])
    upper_color_range = np.array([30, 255,255])


    #create a mask for the color we want to track
    mask = cv2.inRange(hsv_frame, lower_color_range, upper_color_range) 

    #find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Find the largest contour and track it
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 10, (255, 0, 0), -1)  # Draw tracking circle

            #draw a circle around the centroid
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

            #draw a rectangle around the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)  
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #show the frame
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


  
