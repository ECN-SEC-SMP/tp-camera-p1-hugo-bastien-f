import cv2 as cv
import numpy as np

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113
G_KEY = 103
SPACE = 32

# Global variables
chessboardSize = (0, 0)
numImages = 0

# Open camera and set resolution
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

# Capture a frame from the camera and convert to grayscale
def captureFrameFromCamera(cap):
    ret, frame = cap.read() # Capture frame-by-frame

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None, None

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Convert to grayscale

    return frame, gray

# Close camera and destroy all windows
def closeCamera(cap):
    cap.release()
    cv.destroyAllWindows()

# Ask user for camera ID
def askCameraIdToUser():
    print("Select camera to open (enter camera ID, or -1 to exit)")
    try:
        cameraID = int(input())
        return cameraID
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return askCameraIdToUser()

# Ask user for chessboard size
def askChessboardSizeToUser():
    try:
        print("Enter chessboard width")
        width = int(input())

        print("Enter chessboard height")
        height = int(input())

        if width == -1 and height == -1:
            exit()

        global chessboardSize
        chessboardSize = (width, height)
    except ValueError:
        print("Invalid input. Please enter two integers.")
        return askChessboardSizeToUser()
    
# Ask user for number of images to capture
def askNumberOfImagesToUser():
    try:
        print("Enter number of images to capture for calibration (or -1 to exit)")
        num_images = int(input())
        if num_images == -1:
            exit()
        global numImages
        numImages = num_images
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return askNumberOfImagesToUser()

# Detect chessboard corners in the captured frame
def chessBoardDetection(cap):
    frame, gray = captureFrameFromCamera(cap)
    if frame is None:
        return None, None, None

    # Find chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, flags=cv.CALIB_CB_ADAPTIVE_THRESH)

    if ret:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # refine corner locations
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) # refine corner locations
        frame = cv.drawChessboardCorners(frame, (9,6), corners2, ret) # draw corners on frame
        return frame, corners2, gray.shape[::-1]
    else:
        return frame, None, gray.shape[::-1]

# Calibrate camera using object points and image points
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

# Rectify image using calibration parameters
def imageRectification(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    return dst

# Ask user which image to display and show it until ESC/Q
def askImageToDisplay(original_img, rectified_img):
    if original_img is None and rectified_img is None:
        print("No image available to display.")
        return

    while True:
        print("\nWhat do you want to display ?")
        print("1 - Original image")
        print("2 - Rectified image")
        print("3 - Exit display menu")

        try:
            choice = int(input())
        except ValueError:
            print("Invalid input. Please enter 1, 2 or 3.")
            continue

        if choice == 3:
            break
        elif choice == 1:
            if original_img is None:
                print("Original image not available.")
                continue
            window_name = 'original'
            img_to_show = original_img
        elif choice == 2:
            if rectified_img is None:
                print("Rectified image not available. Run calibration first.")
                continue
            window_name = 'rectified'
            img_to_show = rectified_img
        else:
            print("Invalid choice. Please enter 1, 2 or 3.")
            continue

        while True:
            cv.imshow(window_name, img_to_show)
            key = cv.waitKey(30) & 0xFF
            if key in (ESC_KEY, Q_KEY):
                cv.destroyWindow(window_name)
                break

def main():
    counter = 0

    askChessboardSizeToUser()
    askNumberOfImagesToUser()

    # Prepare object points based on chessboard size
    objp = np.zeros((chessboardSize[0]*chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

    objpoints = []  
    imgpoints = [] 

    cap = openCamera()
    last_frame = None
    rectified_img = None
    while True:
        frame, corners, image_size = chessBoardDetection(cap)
        if frame is None:
            break
        last_frame = frame.copy()
        cv.imshow('frame', frame)
        
        key = cv.waitKey(1) & 0xFF
        if corners is not None and key == SPACE:
            objpoints.append(objp)
            imgpoints.append(corners)
            counter += 1

        if key == G_KEY or counter == numImages:
            ret, intrinsic, distCoeffs, rvecs, tvecs = calibrateCamera(objpoints, imgpoints, image_size)
            print("Calibration results:")
            print("Retval:", ret)
            print("Intrinsic Matrix:\n", intrinsic)
            print("Distortion Coefficients:\n", distCoeffs)
            print("Rotation Vectors:\n", rvecs)
            print("Translation Vectors:\n", tvecs)

            rectified_img = imageRectification(frame, intrinsic, distCoeffs)
            break

        if key in (ESC_KEY, Q_KEY):
            break

    closeCamera(cap)

    askImageToDisplay(last_frame, rectified_img)

if __name__ == "__main__":
    main()