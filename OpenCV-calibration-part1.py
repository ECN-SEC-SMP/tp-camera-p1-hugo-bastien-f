import cv2 as cv
import numpy as np

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113
G_KEY = 103

def captureVideoFromCamera(windowName):
    selectedColor = cv.COLOR_BGR2GRAY

    cameraID = askCameraIdToUser()
    if cameraID == -1:
        exit()
    cap = cv.VideoCapture(cameraID)

    while not cap.isOpened():
        print("Cannot open camera")
        cap.release()
        cameraID = askCameraIdToUser()
        if cameraID == -1:
            exit()
        cap = cv.VideoCapture(cameraID)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, selectedColor)

        # Display the resulting frame
        cv.imshow(windowName, gray)

        if cv.waitKey(1) == ESC_KEY:
            break
        elif cv.waitKey(1) == G_KEY:
            if selectedColor == cv.COLOR_BGR2GRAY:
                selectedColor = cv.COLOR_BGR2BGRA
            else:
                selectedColor = cv.COLOR_BGR2GRAY
 
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows(windowName)
    cv.destroyAllWindows()

def playingVideoFromFile(path):
    cap = cv.VideoCapture(path)
 
    while cap.isOpened():
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ESC_KEY:
            break
    
    cap.release()
    cv.destroyAllWindows()

def askCameraIdToUser():
    print("Select camera to open (enter camera ID, or -1 to exit)")
    try:
        cameraID = int(input())
        return cameraID
    except ValueError:
        print("Invalid input. Please enter an integer.")
        askCameraIdToUser()

def main():
    key = None
    windowName = "OpenCV Calibration"
    cv.namedWindow(windowName, cv.WINDOW_AUTOSIZE)

    captureVideoFromCamera(windowName)
    playingVideoFromFile("video_test.mp4")

# Starting the code
if __name__ == "__main__":
    main()