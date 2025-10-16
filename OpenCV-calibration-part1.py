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
            
            # Check if image was loaded successfully
            if image is None:
                print(f"Error: Could not load image {path}")
                return
                
            cv.imshow(window, image)
            # When esc is pressed, exit the loop
            if cv.waitKey(1) == ESC_KEY:
                cleanup(window)
                return

# Starting the code
if __name__ == "__main__":
    main()