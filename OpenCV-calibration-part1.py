# openCV import
import cv2 as cv
import numpy as np

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113

# user must provide a camera device number until one is valid
def openCAM():
    cam_number = input("Enter your cam number : ")
    cam = cv.VideoCapture(int(cam_number))
    if not cam.isOpened():
        while not cam.isOpened():
            cam_number = input("Enter your cam number : ")
            cam = cv.VideoCapture(int(cam_number))
    return cam

def cleanup(window1, window2, cap):
    cv.destroyWindow(window1)
    cv.destroyWindow(window2)
    cap.release()

def main():
    # Open the camera device
    cap = openCAM()
    # Creating the windows to display the video stream
    window1 = "Gray image"
    cv.namedWindow(window1, cv.WINDOW_AUTOSIZE)
    window2 = "Coloured image"
    cv.namedWindow(window2, cv.WINDOW_AUTOSIZE)
    # Boolean value to change between gray and coloured mode
    isGray =  True
    while True:
        # Capture frame-by-frame
        ret, image = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # if g is pressed, change the color mode of win1
        if cv.waitKey(1) == ord('g'):
            isGray = not isGray
        # Our operations on the frame come here
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Display the resulting frame for win 1(coloured or gray depending if g key has been pressed)
        if isGray:
            cv.imshow(window1, gray_image)
        else:
            cv.imshow(window1, image) 
        # Display the resulting frame for win 2
        cv.imshow(window2, image)
        
        # When esc is pressed, exit the loop
        if cv.waitKey(1) == ESC_KEY:
            cleanup(window1, window2, cap)
            break

# Starting the code
if __name__ == "__main__":
    main()