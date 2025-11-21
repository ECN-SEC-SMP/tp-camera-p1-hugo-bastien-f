import glob
import os

import cv2 as cv
import numpy as np

ESC_KEY = 27
Q_KEY = 113


def load_chessboard_images(folder_pattern):
    paths = sorted(glob.glob(folder_pattern))
    images = []
    for p in paths:
        img = cv.imread(p)
        if img is None:
            print(f"Warning: cannot read {p}")
            continue
        images.append((p, img))
    return images


def calibrate_from_images(images, chessboard_size):
    objpoints = []
    imgpoints = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    gray_shape = None

    for path, img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, flags=cv.CALIB_CB_ADAPTIVE_THRESH)
        if not ret:
            print(f"Chessboard not found in {os.path.basename(path)}")
            continue

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)
        gray_shape = gray.shape[::-1]

    if not objpoints:
        print("No valid chessboard detections, calibration aborted.")
        return None

    ret, intrinsic, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        gray_shape,
        None,
        None,
    )

    print("Calibration results from photos:")
    print("Retval:", ret)
    print("Intrinsic Matrix:\n", intrinsic)
    print("Distortion Coefficients:\n", distCoeffs)

    return intrinsic, distCoeffs


def rectify_example_image(img, intrinsic, distCoeffs):
    h, w = img.shape[:2]
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(intrinsic, distCoeffs, (w, h), 1, (w, h))
    dst = cv.undistort(img, intrinsic, distCoeffs, None, newcameramtx)
    return dst


def main():
    chessboard_size = (9, 6)

    images = load_chessboard_images("calib_gopro/*.jpg")
    if not images:
        print("No images found in calib_gopro folder.")
        return

    intrinsic, distCoeffs = calibrate_from_images(images, chessboard_size)
    if intrinsic is None:
        return

    sample_img = images[0][1]
    rectified = rectify_example_image(sample_img, intrinsic, distCoeffs)

    while True:
        cv.imshow("original", sample_img)
        cv.imshow("rectified", rectified)
        key = cv.waitKey(30) & 0xFF
        if key in (ESC_KEY, Q_KEY):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
