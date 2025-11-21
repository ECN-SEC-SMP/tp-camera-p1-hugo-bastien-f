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

def cleanup(window):
    cv.destroyWindow(window)

def main():
    path = "../calib_gopro/calib_gopro/GOPR84"
    # Creating the windows to display the video stream
    window = "Gopro calib"
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    # Boolean value to change between gray and coloured mode
    isGray =  True
    while True:
        # Capture frame-by-frame
        for i in range(1, 27):
            # Format the path correctly with zero-padding
            path = f"../calib_gopro/calib_gopro/GOPR84{i:02d}.JPG"
            image = cv.imread(path)
            if image is None:
                print(f"Error: Could not load image {path}")
                exit(0)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
            # Check if image was loaded successfully
            if image is None:
                print(f"Error: Could not load image {path}")
                return
            ret, corners = cv.findChessboardCorners(image, (9,6), flags=cv.CALIB_CB_ADAPTIVE_THRESH)
            if ret == True:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                cv.drawChessboardCorners(image, (7,6), corners2, ret)    
                cv.imshow(window, image)
            else:
                cv.imshow(window, image)
            # Wait for a key press and exit if 'esc' is pressed
            if cv.waitKey(0) == ESC_KEY:
                cleanup(window)
                return

# Starting the code
if __name__ == "__main__":
    main()