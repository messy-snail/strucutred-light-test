import copy

import cv2
from cv2 import aruco
import numpy as np

charucoCornersAccum =[]
charucoIdsAccum =[]
def intersect_circles_rays2board(circles, rvec, t, K, dist_coef):
    circles_normalized = cv2.convertPointsToHomogeneous(cv2.undistortPoints(circles, K, dist_coef))
    if not rvec.size:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    plane_normal = R[2,:] # last row of plane rotation matrix is normal to plane
    plane_point = t.T     # t is a point on the plane
    epsilon = 1e-06
    circles_3d = np.zeros((0,3), dtype=np.float32)
    for p in circles_normalized:
        ray_direction = p / np.linalg.norm(p)
        ray_point = p
        ndotu = plane_normal.dot(ray_direction.T)
        if abs(ndotu) < epsilon:
            print ("no intersection or line is within plane")
        w = ray_point - plane_point
        si = -plane_normal.dot(w.T) / ndotu
        Psi = w + si * ray_direction + plane_point
        circles_3d = np.append(circles_3d, Psi, axis = 0)
    return circles_3d


# --------- detect ChAruco board -----------
# arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
# arucoParams = cv2.aruco.DetectorParameters_create()
# (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
# 	parameters=arucoParams)
def detect_aruco(img_list):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    K = np.load('camera_matrix.npy')
    dist_coef = np.load('distortion.npy')

    proj_circles_list= []
    cam_circles_list = []
    circles_3d_list = []

    # circles_3d_list = np.zeros((5 * 4 * 8, 3), np.float32)

    # img_idx =0
    # num_total_imgs = len(img_list)
    for input in img_list:

        frame = copy.deepcopy(input)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_inv = cv2.bitwise_not(gray)

        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        cb = aruco.CharucoBoard_create(10, 14, 20, 14, aruco_dict)

        corners, ids, rejected = cv2.aruco.detectMarkers(frame, cb.dictionary)
        if ids is None:
            continue
        corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(frame, cb, corners, ids, rejected, cameraMatrix=K, distCoeffs=dist_coef)
        # if corners == None or len(corners) == 0:`
        #     continue
        ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, cb)

        if ret==False:
            continue
        aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0,255,0))

        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, cb, K, dist_coef, None, None)
        # // if charuco pose is valid
        if valid:
            aruco.drawAxis(frame, K, dist_coef, rvec, tvec, 25)


        circles_grid_size = (4, 5)
        for corner in corners:
            gray_inv = cv2.fillConvexPoly(gray_inv, corner.astype(np.int32), (255))

        try:
            ret, circles = cv2.findCirclesGrid(gray_inv, circles_grid_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        except Exception as e:
            print(e)
            return None
        if ret ==False:
            continue
            # return None

        cam_circles_list.append(circles)
        img = cv2.drawChessboardCorners(frame, circles_grid_size, circles, ret)
        # ray-plane intersection: circle-center to chessboard-plane
        circles3D = intersect_circles_rays2board(circles, rvec, tvec, K, dist_coef)


        if circles3D is None:
            return None
        # re-project on camera for verification
        circles3D_reprojected, _ = cv2.projectPoints(circles3D, (0, 0, 0), (0, 0, 0), K, dist_coef)
        # circles3D_reprojected, _ = cv2.projectPoints(circles3D, rvec, tvec, K, dist_coef)


        # for i in range(0, len(circles3D)):
        #     circles_3d_list[i+img_idx*num_total_imgs] = circles_3d_list[i]
        circles_3d_list.append(circles3D)
        proj_circles_list.append(circles3D_reprojected.squeeze())

        for c in circles3D_reprojected:
            cv2.circle(img, tuple(c.astype(np.int32)[0]), 3, (255, 255, 0), cv2.FILLED)

        # img_idx+=1
        print('pass')
    # # calibrate projector
    w_proj = 1280
    h_proj = 800

    K_proj = np.array([[1000., 0., w_proj / 2.],
                                 [0., 1000., h_proj / 2.],
                                 [0., 0., 1.]])

    dist_coef_proj = np.zeros((5, 1))

    print("calibrate projector")
    print("proj calib mat before\n%s" % K_proj)

    #objectPointsAccum: 실제 원 크기
    #projCirclePoints 찾은 값

    ret, K_proj, dist_coef_proj, rvecs, tvecs = cv2.calibrateCamera(np.array(circles_3d_list, np.float32),
                                                                    np.array(proj_circles_list, np.float32),
                                                                    (w_proj, h_proj),
                                                                    K_proj,
                                                                    dist_coef_proj,
                                                                    flags=cv2.CALIB_USE_INTRINSIC_GUESS)


    print("proj calib mat after\n%s" % K_proj)
    print("proj dist_coef %s" % dist_coef_proj.T)
    print("calibration reproj err %s" % ret)
    print("stereo calibration")

    w_cam = 1280
    h_cam = 960

    ret, K, dist_coef, K_proj, dist_coef_proj, proj_R, proj_T, _, _ = cv2.stereoCalibrate(
        np.array(circles_3d_list, np.float32),
        np.array(cam_circles_list, np.float32),
        np.array(proj_circles_list, np.float32),
        K,
        dist_coef,
        K_proj,
        dist_coef_proj,
        (w_cam, h_cam),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    proj_rvec, _ = cv2.Rodrigues(proj_R)
    print("R \n%s" % proj_R)
    print("T %s" % proj_T.T)
    print("proj calib mat after\n%s" % K_proj)
    print("proj dist_coef %s" % dist_coef_proj.T)
    print("cam calib mat after\n%s" % K)
    print("cam dist_coef %s" % dist_coef.T)
    print("reproj err %f" % ret)

    return img


def calibrate_camera(img_list):
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    cb = aruco.CharucoBoard_create(10, 14, 20, 14, aruco_dict)
    imsize = img_list[0].shape[:2]
    for im in img_list:
        frame = im
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, cb)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator += 1

    cameraMatrixInit = np.array([[1000., 0., imsize[0] / 2.],
                                 [0., 1000., imsize[1] / 2.],
                                 [0., 0., 1.]])

    distCoeffsInit = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)

    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=cb,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    np.save('distortion', distortion_coefficients0)  # x_save.npy
    np.save('camera_matrix', camera_matrix)  # x_save.npy


    pass
# charucoCornersAccum += [charucoCorners]
# charucoIdsAccum += [charucoIds]
#
# K = np.array([[1000., 0., imsize[0] / 2.],
#                              [0., 1000., imsize[1] / 2.],
#                              [0., 0., 1.]])
#
# dist_coef = np.zeros((5, 1))
#
# if number_charuco_views == 40:
#     print("calibrate camera")
#     print("camera calib mat before\n%s"%K)
#     # calibrate camera
#     ret, K, dist_coef, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(charucoCornersAccum,
#                                                                        charucoIdsAccum,
#                                                                        cb,
#                                                                        (w, h),
#                                                                        K,
#                                                                        dist_coef,
#                                                                        flags = cv2.CALIB_USE_INTRINSIC_GUESS)
#     print("camera calib mat after\n%s"%K)
#     print("camera dist_coef %s"%dist_coef.T)
#     print("calibration reproj err %s"%ret)
#
# # --------- detect circles -----------
# ret, circles = cv2.findCirclesGrid(gray, circles_grid_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
# img = cv2.drawChessboardCorners(img, circles_grid_size, circles, ret)
# # ray-plane intersection: circle-center to chessboard-plane
# circles3D = intersectCirclesRaysToBoard(circles, rvec, tvec, K, dist_coef)
# # re-project on camera for verification
# circles3D_reprojected, _ = cv2.projectPoints(circles3D, (0,0,0), (0,0,0), K, dist_coef)
# for c in circles3D_reprojected:
#     cv2.circle(img, tuple(c.astype(np.int32)[0]), 3, (255,255,0), cv2.FILLED)