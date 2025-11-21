# openCV import
import cv2 as cv
import numpy as np

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113
SPACE_KEY = 32

# user must provide a camera device number until one is valid
def openCAM():
    cam_number = input("Enter your cam number : ")
    cam = cv.VideoCapture(int(cam_number))
    if not cam.isOpened():
        while not cam.isOpened():
            cam_number = input("Enter your cam number : ")
            cam = cv.VideoCapture(int(cam_number))
    return cam

def findNDisplayChessBoardCorners(image, pattern_size=(9,6), objpoints=None, imgpoints=None):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Check for chessboard corners
    ret, corners = cv.findChessboardCorners(gray, pattern_size, flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    # If found, refine corners and return image with drawn corners
    if ret == True:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv.drawChessboardCorners(image, pattern_size, corners2, ret)
        return image, ret, corners2
    # If not found, return original image
    else:
        return image, ret, None

def cleanup(window, cap):
    cv.destroyWindow(window)
    cap.release()

def main():
    # Prepare object points for 9x6 chessboard
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Open the camera device
    cap = openCAM()
    # Creating the windows to display the video stream
    window = "image"
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    
    print("Press SPACE to capture an image for calibration")
    print("Press ESC to finish and calibrate")
    
    capture_count = 0
    
    while True:
        # Capture frame-by-frame
        ret, image = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Check if image was loaded successfully
        if image is None:
            print(f"Error: Could not load image")
            return
        
        # Check for chessboard corners and display them
        result, found, corners = findNDisplayChessBoardCorners(image)
        
        # Display capture count
        cv.putText(result, f"Captures: {capture_count}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv.imshow(window, result)
        
        key = cv.waitKey(1)
        
        # When SPACE is pressed, save the corners if found
        if key == SPACE_KEY and found:
            objpoints.append(objp)
            imgpoints.append(corners)
            capture_count += 1
            print(f"Image {capture_count} captured!")
        
        # When ESC is pressed, calibrate and exit
        if key == ESC_KEY:
            if len(objpoints) > 0:
                print("Calibrating camera...")
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
                
                print("\nCalibration completed!")
                print(f"Camera matrix:\n{mtx}")
                print(f"\nDistortion coefficients:\n{dist}")
            else:
                print("No images captured for calibration!")
            
            cleanup(window, cap)
            break

# Starting the code
if __name__ == "__main__":
    main()