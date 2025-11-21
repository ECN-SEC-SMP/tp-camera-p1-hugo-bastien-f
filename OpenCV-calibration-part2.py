import cv2 as cv
import numpy as np

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113
G_KEY = 103
SPACE = 32

# Chessboard pattern size (number of inner corners)
CHESSBOARD_SIZE = (9, 6)

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

def chessBoardDetection(cap):
    frame, gray = captureFrameFromCamera(cap)
    if frame is None:
        return None, None, None

    ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    if ret:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        frame = cv.drawChessboardCorners(frame, (9,6), corners2, ret)
        return frame, corners2, gray.shape[::-1]
    else:
        return frame, None, gray.shape[::-1]


def calibrateCamera(objpoints, imgpoints, image_size):
    if len(objpoints) == 0 or len(imgpoints) == 0:
        return None, None, None, None, None

    ret, intrinsic, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )
    return ret, intrinsic, distCoeffs, rvecs, tvecs

def main():
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    objpoints = []  
    imgpoints = [] 

    cap = openCamera()
    while True:
        frame, corners, image_size = chessBoardDetection(cap)
        if frame is None:
            break
        cv.imshow('frame', frame)
        
        key = cv.waitKey(1) & 0xFF
        if corners is not None and key == SPACE:
            objpoints.append(objp)
            imgpoints.append(corners)

        if key == G_KEY:
            ret, intrinsic, distCoeffs, rvecs, tvecs = calibrateCamera(objpoints, imgpoints, image_size)
            print("Calibration results:")
            print("Retval:", ret)
            print("Intrinsic Matrix:\n", intrinsic)
            print("Distortion Coefficients:\n", distCoeffs)
            print("Rotation Vectors:\n", rvecs)
            print("Translation Vectors:\n", tvecs)

        if key in (ESC_KEY, Q_KEY):
            return None

# Starting the code
if __name__ == "__main__":
    main()