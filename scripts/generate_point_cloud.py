#!/usr/bin/env python3

"""
 * @copyright Copyright (c) 2023
 * @file base.py
 * @author Kshitij Karnawat (kshitij@umd.edu)
 * @author Hritvik Choudhari (hac@umd.edu)
 * @brief 
 * @version 0.1
 * @date 2023-04-17
 * 
 * 
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import open3d as o3d
import glob


def getMatches(image1, image2, number):
    """
        Initialize a new Contact object.

        Args:
            image1 (cv::Mat): The first image
            image2 (cv::Mat): The second image
            number (int): Number of matches to consider

        Returns:
            points1 (numpy array, int32): Feature Points for first image
            points2 (numpy array, int32): Feature Points for second image

    """
    sift = cv.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(image1,None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2,None)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key = lambda x:x.distance)
  
    print("No. of matching features found =",len(matches))
    points1 = []
    points2 = []

    for i in matches[:number]:
        x1, y1 = keypoints1[i.queryIdx].pt
        x2, y2 = keypoints2[i.trainIdx].pt
        points1.append([x1, y1])
        points2.append([x2, y2])

    image1_copy = image1.copy()
    image2_copy = image2.copy()

    image1_features = cv.drawKeypoints(image1, keypoints1, image1_copy)
    image2_features = cv.drawKeypoints(image2, keypoints2, image2_copy)

    draw_images = cv.drawMatches(image1, keypoints1, image2, keypoints2, matches[:number], image2_copy, flags=2)

    cv.imshow("image1_features", image1_features)
    cv.waitKey(0)

    cv.imshow("image2_features", image2_features)
    cv.waitKey(0)

    cv.imshow("Keypoint matches", draw_images)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return np.array(points1, dtype = np.int32), np.array(points2, dtype = np.int32)


def getDispartiy(image1, image2, h, w):
    """
        Initialize a new Contact object.

        Args:
            image1 (cv::Mat): The first image
            image2 (cv::Mat): The second image
            h (int): height of image
            w (int): width of image

        Returns:
            disparity (image matrix): disparity of the stereo image

    """

    disparity = np.zeros((h,w), np.uint8)
    image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    stereo = cv.StereoSGBM_create(1,
                                  128,
                                  3,
                                speckleRange=1)
                                # speckleWindowSize=50)
    disparity = stereo.compute(image1_gray, image2_gray)

    return disparity


def getDepth(disparity, baseline, focul_length):

    depth = (baseline * focul_length) / (disparity + 1e-10)
    depth[depth > 10000] = 10000
    depth_map = np.uint8(depth * 255 / np.max(depth))

    return depth_map


def main():
    # pcds= []
    print("loading images")
    
    image1 = cv.imread("../Data/rotate/left/IMG_20230516_234023.jpg")
    image2 = cv.imread("../Data/rotate/right/IMG_20230516_234025.jpg")

    image1 = cv.resize(image1, (int(image1.shape[1] * 0.5), int(image1.shape[0] * 0.5)), interpolation=cv.INTER_AREA)
    image2 = cv.resize(image2, (int(image2.shape[1] * 0.5), int(image2.shape[0] * 0.5)), interpolation=cv.INTER_AREA)

    print("images loaded")

    intrinsic_matrix = np.array([[1552.58610, 0, 768.938827],
                                 [0, 1552.57499, 1006.09656],
                                 [0, 0, 1]])


    points1, points2 = getMatches(image1, image2, 500)

    fundamental_matrix, _ = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, 0.5, 0.99)
    print("\nF = ", fundamental_matrix )
    count = 0
    for i in range(len(_)):
        if _[i] == 1:
            count+=1
    print(count)
    # essential_matrix = cv.findEssentialMat(points1, points2, intrinsic_matrix, cv.RANSAC, 0.99, 0.1)

    h1, w1 = image1.shape[:2]

    _, H1, H2 = cv.stereoRectifyUncalibrated(points1, points2, fundamental_matrix, imgSize=(w1, h1))
    
    print("\nH1 =", H1) 
    print("\nH2 =", H2)

    disparity = getDispartiy(image1, image2, image1.shape[0], image1.shape[1])
    plt.imshow(disparity)
    plt.show()
    
    disparity = disparity/16

    pcd = []
    height, width = disparity.shape[0], disparity.shape[1]
    for i in range(height):
        for j in range(width):
            z = disparity[i][j]
            x = (j - intrinsic_matrix[0,2]) * z / intrinsic_matrix[0,0]
            y = (i - intrinsic_matrix[1,2]) * z / intrinsic_matrix[1,1]
            pcd.append([x, y, z])

    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    # Visualize:
    o3d.visualization.draw_geometries([pcd_o3d])

    # ## TODO: Create a function for the code below 
    # # Point Cloud Generation from disparity(depth) and RGB Image
    # image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)

    # depth = np.float32(depth/16) # As mentioned in documentation (Required step)

    # # Convert Depth to Open3D Image Format
    # depth_as_img = o3d.geometry.Image((np.ascontiguousarray(depth)).astype(np.float32))    
    
    # # Intrinsic Parameter in Open3D Format
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(image1.shape[1], image1.shape[0], intrinsic_matrix[0,0], intrinsic_matrix[0,0], intrinsic_matrix[0,2], intrinsic_matrix[1,2])
   
    # # Generte RGBD Image
    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(image1), depth_as_img, convert_rgb_to_intensity=False)
    
    # # Generate Point Cloud
    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
   
    # # flip the orientation, so it looks upright, not upside-down
    # pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    
    # # pcds.append(pcd)
    
    # ## TODO: Stitch Multiple Point Clouds to construct Model

    # # Point Cloud Visualization
    # o3d.visualization.draw_geometries([pcd]) 


if __name__ == "__main__":
    main()