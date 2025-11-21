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

def findNDisplayChessBoardCorners(image, pattern_size=(9,6)):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Check for chessboard corners
    ret, corners = cv.findChessboardCorners(image, pattern_size, flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    # If found, refine corners and return image with drawn corners
    if ret == True:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv.drawChessboardCorners(image, (9,6), corners2, ret)   
        return image
    # If not found, return original image
    else:
        return image

def cleanup(window, cap):
    cv.destroyWindow(window)
    cap.release()

def main():
    # Open the camera device
    cap = openCAM()
    # Creating the windows to display the video stream
    window = "image"
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    while True:
        # Capture frame-by-frame
        ret, image = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Display the resulting frame
        # Check if image was loaded successfully
        if image is None:
            print(f"Error: Could not load image")
            return
        # Check for chessboard corners and display them
        result = findNDisplayChessBoardCorners(image)
        cv.imshow(window, result)
        # When esc is pressed, exit the loop
        if cv.waitKey(1) == ESC_KEY:
            cleanup(window, cap)
            break

# Starting the code
if __name__ == "__main__":
    main()