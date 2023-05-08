import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import open3d as o3d


def getMatches(image1, image2, number):
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

    # cv.imshow("image1_features", image1_features)
    # cv.waitKey(0)

    # cv.imshow("image2_features", image2_features)
    # cv.waitKey(0)

    # cv.imshow("Keypoint matches", draw_images)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return np.array(points1, dtype = np.int32), np.array(points2, dtype = np.int32)


def getDispartiy(image1, image2, h, w):
        disparity = np.zeros((h,w), np.uint8)
        image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
        image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
        stereo = cv.StereoSGBM_create(2,
                                      256,
                                      21,
                                      speckleRange=2,
                                      speckleWindowSize=200)
        disparity = stereo.compute(image1_gray, image2_gray)

        return disparity


def main():
    image1 = cv.imread("../chess/im0.png")
    image2 = cv.imread("../chess/im1.png")

    intrinsic_matrix = np.array([[1733.74, 0, 792.27],
                                        [0, 1733.74, 541.89],
                                        [0, 0, 1]])

    points1, points2 = getMatches(image1, image2, 100)

    fundamental_matrix, _ = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, 0.1, 0.99)

    essential_matrix = cv.findEssentialMat(points1, points2, intrinsic_matrix, cv.RANSAC, 0.99, 0.1)
    # print("points2", fundamental_matrix )


    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    _, H1, H2 = cv.stereoRectifyUncalibrated(points1, points2, fundamental_matrix, imgSize=(w1, h1))
    
    # print("\nH1 =", H1) 
    # print("\nH2 =", H2)

    disparity = getDispartiy(image1, image2, image1.shape[0], image1.shape[1])
    plt.imshow(disparity, cmap='jet', interpolation='gaussian')
    plt.show()

    disparity = np.float32(disparity/16)

    depth_as_img = o3d.geometry.Image((np.ascontiguousarray(disparity)).astype(np.float32))    
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(image1.shape[1], image1.shape[0], intrinsic_matrix[0,0], intrinsic_matrix[0,0], intrinsic_matrix[0,2], intrinsic_matrix[1,2])
    image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(image1), depth_as_img, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images, camera, option)

    # flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    
    o3d.visualization.draw_geometries([pcd]) 

if __name__ == "__main__":
    main()