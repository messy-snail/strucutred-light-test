import cv2


# --------- detect ChAruco board -----------
corners, ids, rejected = cv2.aruco.detectMarkers(frame, cb.dictionary)
corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(frame, cb, corners, ids, rejected, cameraMatrix=K, distCoeffs=dist_coef)
if corners == None or len(corners) == 0:
    continue
ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, cb)
charucoCornersAccum += [charucoCorners]
charucoIdsAccum += [charucoIds]
if number_charuco_views == 40:
    print("calibrate camera")
    print("camera calib mat before\n%s"%K)
    # calibrate camera
    ret, K, dist_coef, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charucoCornersAccum,
                                                                       charucoIdsAccum,
                                                                       cb,
                                                                       (w, h),
                                                                       K,
                                                                       dist_coef,
                                                                       flags = cv2.CALIB_USE_INTRINSIC_GUESS)
    print("camera calib mat after\n%s"%K)
    print("camera dist_coef %s"%dist_coef.T)
    print("calibration reproj err %s"%ret)