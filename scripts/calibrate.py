import cv2
import numpy as np
import os
import glob

world_coords= [(0,0,0),(22.5,0,0),(45,0,0),(67.5,0,0),(90,0,0),(112.5,0,0),(129,0,0),(150.5,0,0),(172,0,0),
              (0,22.5,0),(22.5,22.5,0),(45,22.5,0),(67.5,22.5,0),(90,22.5,0),(112.5,22.5,0),(129,22.5,0),(150.5,22.5,0),(172,22.5,0),
              (0,45,0),(22.5,45,0),(45,45,0),(67.5,45,0),(90,45,0),(112.5,45,0),(129,45,0),(150.5,45,0),(172,45,0),
              (0,67.5,0),(22.5,67.5,0),(45,67.5,0),(67.5,67.5,0),(90,67.5,0),(112.5,67.5,0),(129,67.5,0),(150.5,67.5,0),(172,67.5,0),
               (0,90,0),(22.5,90,0),(45,90,0),(67.5,90,0),(90,90,0),(112.5,90,0),(129,90,0),(150.5,90,0),(172,90,0),
               (0,112.5,0),(22.5,112.5,0),(45,112.5,0),(67.5,112.5,0),(90,112.5,0),(112.5,112.5,0),(129,112.5,0),(150.5,112.5,0),(172,112.5,0)]

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
path = r'../Data/Calibration images'
world_coords = np.array(world_coords, dtype= np.float32)
world_points = []
img_points = []
image_frames = glob.glob(os.path.join(path,'*.png'))
for i in image_frames:
    img = cv2.imread(i)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_img, (9,6), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        world_points.append(world_coords)
        corners_mod = cv2.cornerSubPix(gray_img,corners, (11,11), (-1,-1), criteria)
        img_points.append(corners_mod)
        cv2.drawChessboardCorners(img, (9,6), corners_mod, ret)
        resized_img= cv2.resize(img,(1920,1080),interpolation= cv2.INTER_AREA)
        cv2.imshow('img', resized_img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

ret, K, dist, R_vec, T_vec = cv2.calibrateCamera(world_points, img_points, gray_img.shape[::-1], None, None)
per_frame_rep_error = []
mean_error = 0
for i in range(len(world_points)):
    pred_imgpoints, _ = cv2.projectPoints(world_points[i], R_vec[i], T_vec[i], K, dist)
    error = cv2.norm(img_points[i], pred_imgpoints, cv2.NORM_L2)/len(pred_imgpoints)
    per_frame_rep_error.append(error)
    mean_error += error
frame= 1
for err in per_frame_rep_error:
    print("Reprojection error for {} frame is: {}".format(frame,err))
    frame+=1
print("*********************************************")
print( "Average reprojection error across all frames: {}".format(mean_error/len(world_points)))
print("*********************************************")
print("The intrinsic matrix (K) is: {}".format(K))