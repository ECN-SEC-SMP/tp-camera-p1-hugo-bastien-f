import cv2 as cv
import numpy as np

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113

def captureVideoFromCamera(cameraID):
    cap = cv.VideoCapture(cameraID)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ESC_KEY:
            break
 
    # When everything done, release the capture
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

def main():
    key = None
    captureVideoFromCamera(0)
    playingVideoFromFile("video_test.mp4")

# Starting the code
if __name__ == "__main__":
    main()