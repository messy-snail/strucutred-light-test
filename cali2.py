import numpy as np
import cv2
import glob
import sys

###################
'''This code is for performing the stereo calibration if 
the intrinsic parameters and the necessary images are available.'''
###################
# Create ChArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard_create(9, 6, 3, 1.25, aruco_dict)
arucoParams = cv2.aruco.DetectorParameters_create()

counter_L, corners_list_L, id_list_L = [], [], []
counter_R, corners_list_R, id_list_R = [], [], []

# load the intrinsic parameters obtained from Int_Calibration.py
with np.load('c2.npz') as X:
    mtxR, distR, rvecsR, tvecsR = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

with np.load('c1.npz') as X:
    mtxL, distL, rvecsL, tvecsL = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpointsL = []  # 2d points in left image plane
imgpointsR = []  # 2d points in right image plane

# Get images for left and right directory
images_l = glob.glob('data1/*.jpg')
images_r = glob.glob('data2/*.jpg')

# Images should be perfect pairs. Otherwise all the calibration will be false
images_l.sort()
images_r.sort()

# Pairs should be same size. Otherwise we have sync problem.
if len(images_l) != len(images_r):
    print("Numbers of left and right images are not equal. They should be pairs.")
    print("Left images count: ", len(images_l))
    print("Right images count: ", len(images_r))
    sys.exit(-1)

# Pair the images for single loop handling
pair_images = zip(images_l, images_r)

for images_l, images_r in pair_images:
    # Left Image Points
    img_l = cv2.imread(images_l)
    grayL = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    corners_L, ids_L, rejected_L = cv2.aruco.detectMarkers(grayL, aruco_dict, parameters=arucoParams)
    resp_L, charuco_corners_L, charucos_ids_L = cv2.aruco.interpolateCornersCharuco(corners_L, ids_L, grayL, board)

    # Right Image Points
    img_r = cv2.imread(images_r)
    grayR = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    corners_R, ids_R, rejected_R = cv2.aruco.detectMarkers(grayR, aruco_dict, parameters=arucoParams)
    resp_R, charuco_corners_R, charucos_ids_R = cv2.aruco.interpolateCornersCharuco(corners_R, ids_R, grayR, board)


    objpoints_L, imgpoints_L = cv2.aruco.getBoardObjectAndImagePoints(board, charuco_corners_L, charucos_ids_L)
    objpoints_R, imgpoints_R = cv2.aruco.getBoardObjectAndImagePoints(board, charuco_corners_R, charucos_ids_R)

    if resp_L == resp_R and (resp_L and resp_R) > 1:
        corners_list_L.append(charuco_corners_L)
        corners_list_R.append(charuco_corners_R)

        id_list_L.append(charucos_ids_L)
        id_list_R.append(charucos_ids_R)

        objpoints.append(objpoints_L)
        imgpointsR.append(imgpoints_R)
        imgpointsL.append(imgpoints_L)
        # Draw and display the corners
        cv2.aruco.drawDetectedCornersCharuco(img_l, charuco_corners_L, charucos_ids_L, (255,0,0))
        cv2.aruco.drawDetectedCornersCharuco(img_r, charuco_corners_R, charucos_ids_R, (255,0,0))

        cv2.imshow('imgL', img_l)
        cv2.imshow('imgR', img_r)
        cv2.moveWindow("imgR", 800, 0)
        cv2.waitKey(200)
    else:
        print("Chessboard couldn't detected. Image pair: ", images_l, " and ", images_r)
        continue

cv2.destroyAllWindows()

ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR,(640, 480))
print(T)