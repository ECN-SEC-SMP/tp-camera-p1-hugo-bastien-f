import cv2 as cv
import numpy as np

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113
G_KEY = 103

def openCamera():
    cameraID = askCameraIdToUser()
    if cameraID == -1:
        exit()
    cap = cv.VideoCapture(cameraID)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while not cap.isOpened():
        print("Cannot open camera")
        cap.release()
        cameraID = askCameraIdToUser()
        if cameraID == -1:
            exit()
        cap = cv.VideoCapture(cameraID)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    return cap

def captureFrameFromCamera(cap):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None, None

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    return frame, gray
 
def closeCamera(cap):
    cap.release()
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
        return askCameraIdToUser()

def chessBoardDetection():
    cap = openCamera()
    while True:
        frame, gray = captureFrameFromCamera(cap)
        if frame is None:
            break

        ret, corners = cv.findChessboardCorners(gray, (9,6), flags=cv.CALIB_CB_ADAPTIVE_THRESH)

        if ret:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            frame = cv.drawChessboardCorners(frame, (9,6), corners2, ret)

        cv.imshow('frame', frame)

        key = cv.waitKey(1) & 0xFF
        if key in (ESC_KEY, Q_KEY):
            break

    closeCamera(cap)

def main():
    chessBoardDetection()

# Starting the code
if __name__ == "__main__":
    main()