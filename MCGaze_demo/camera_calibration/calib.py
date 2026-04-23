import numpy as np
import cv2 as cv
import glob
import pyrealsense2 as rs

import pyrealsense2 as rs
import numpy as np
import cv2

IMAGE_DIR = "/home/kovan/FaceAndGaze/MCGaze_demo/camera_calibration/img"
GRID_SHAPE = (6,8)

def photo():
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream (adjust resolution if needed)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start pipeline
    pipeline.start(config)

    i = 1
    while True:
        # Capture frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue  # Skip if no frame available

        # Convert frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Display the image
        cv2.imshow("Color Frame", color_image)

        # Save image on key press
        key = cv2.waitKey(1)
        fname = f"{IMAGE_DIR}-{i}.png"
        if key == ord('s'):  # Press 's' to save
            cv2.imwrite(fname, color_image)
            print(f"Image saved as {fname}")
            i +=1
        elif key == ord('q'):  # Press 'q' to exit
            break

    # Stop pipeline and close window
    pipeline.stop()
    cv2.destroyAllWindows()
    
def project_error():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((GRID_SHAPE[0]*GRID_SHAPE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:GRID_SHAPE[0],0:GRID_SHAPE[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(IMAGE_DIR + "/*.png")

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, GRID_SHAPE, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, GRID_SHAPE, corners2, ret)
            cv.imshow('img', img)
            cv2.imwrite(fname[:-4] + "-pattern" + ".png", img)
            cv.waitKey(500)

    cv.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("mtx: ", mtx)
    print("dist: ", dist)
    print("rvecs: ", rvecs)
    print("tvecs: ", tvecs)
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    print( "mean error: {}".format(mean_error) )

if __name__ == "__main__":
    # photo()
    project_error()
        
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((9*7,3), np.float32)
# objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.

# images = glob.glob('*.jpg')

# for fname in images:
#     img = cv.imread(fname)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, (7,6), None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)

#         corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)

#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (7,6), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(500)

# cv.destroyAllWindows()