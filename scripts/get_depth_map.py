import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

""" 
gstreamer_pipeline object that returns a GStreamer pipeline for capturing the CSI camera and 
Flip the image by setting the flip_method display_width and display_height determine the size 
of each camera pane in the window on the screen
"""
# Gstreamer pipeline for left camera
def gstreamer_pipeline_left(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Gstreamer pipeline for right camera
def gstreamer_pipeline_right(
    sensor_id=1,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Function to get matches in image pairs
def getMatches(image1, image2, number, detector):
    keypoints1, descriptors1 = detector.detectAndCompute(image1,None)
    keypoints2, descriptors2 = detector.detectAndCompute(image2,None)

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

    return np.array(points1, dtype = np.int32), np.array(points2, dtype = np.int32)


# Function to get the disparity map using stereo SGBM function
def getDispartiy(image1, image2, h, w):

    image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    #stereo = cv.StereoSGBM_create(1, 128, 3, speckleRange=1)
    stereo = cv.StereoSGBM_create(1, 64, 11, speckleRange=1)
    #stereo = cv.StereoBM_create(numDisparities=64, blockSize=7)
                                # speckleWindowSize=50)
    disparity = stereo.compute(image1_gray, image2_gray)
    disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    print(np.max(disparity), np.min(disparity))

    return disparity




def main():
    
    window_title_left = "CSI Camera left"
    window_title_right = "CSI Camera right"
    filter_img = True
    dil_kernel = np.ones((3, 3), np.uint8)
    #kernel2 = np.ones((3, 3), np.float32) / 9
    j = 0
    kernel2 = np.ones((7, 7), np.float32) / 49

    '''
    intrinsic_matrix = np.array([[1552.58610, 0, 768.938827],
                                 [0, 1552.57499, 1006.09656],
                                 [0, 0, 1]])

    '''
    intrinsic_matrix = np.array([[1337.89758, 0, 440.149059],
                                 [0, 1338.17020, 299.853520],
                                 [0, 0, 1]])
    

    # Create ORB feature detector
    orb_detector = cv.ORB_create()

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline_left(flip_method=0))

    # Create camera capture objects for left and right cameras
    video_capture_left = cv.VideoCapture(gstreamer_pipeline_left(flip_method=2), cv.CAP_GSTREAMER)
    video_capture_right = cv.VideoCapture(gstreamer_pipeline_right(flip_method=2), cv.CAP_GSTREAMER)

    # Wait for cameras to open
    if video_capture_left.isOpened() and video_capture_right.isOpened():
        try:
            window_handle_left = cv.namedWindow(window_title_left, cv.WINDOW_AUTOSIZE)
            window_handle_right = cv.namedWindow(window_title_right, cv.WINDOW_AUTOSIZE)
            while True:
                while j < 20:
                    ret_val_left, frame_left = video_capture_left.read()
                    ret_val_right, frame_right = video_capture_right.read()
                    j += 1

                ret_val_left, frame_left = video_capture_left.read()
                ret_val_right, frame_right = video_capture_right.read()
                #frame_left = cv.flip(frame_left, 1)                
                #frame_right = cv.flip(frame_right, 1)
                if filter_img == True:
                    frame_left = cv.filter2D(src=frame_left, ddepth=-1, kernel=kernel2)
                    frame_right = cv.filter2D(src=frame_right, ddepth=-1, kernel=kernel2)
                
                # Calculating the fundamental matrix
                points1, points2 = getMatches(frame_left, frame_right, 1000, detector=orb_detector)

                # Calculating the fundamental matrix and essential matrix
                fundamental_matrix, _ = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, 0.5, 0.99)
                essential_matrix = cv.findEssentialMat(points1, points2, intrinsic_matrix, cv.RANSAC, 0.99, 0.1)

                h1, w1 = frame_left.shape[:2]

                # Find the homographies
                _, H1, H2 = cv.stereoRectifyUncalibrated(points1, points2, fundamental_matrix, imgSize=(w1, h1))
                
                #print("\nH1 =", H1) 
                #print("\nH2 =", H2)

                # Get disparity map from left and right images
                disparity = getDispartiy(frame_left, frame_right, frame_left.shape[0], frame_left.shape[1])

                # Trying out different blurring and filtering techniques to improve the quality of disparity maps

                #disparity = cv.dilate(disparity, dil_kernel, iterations=1)
                disparity = cv.medianBlur(disparity, 7)
                #disparity = cv.filter2D(src=disparity, ddepth=-1, kernel=kernel2)
                #disparity = cv.filter2D(src=disparity, ddepth=-1, kernel=kernel2)

                # Show the disparity map
                cv.imshow("disparity map obtained", disparity)

                # Display the left and right frames
                if cv.getWindowProperty(window_title_left, cv.WND_PROP_AUTOSIZE) >= 0:
                    cv.imshow(window_title_left, frame_left)
                    cv.imshow(window_title_right, frame_right)
                else:
                    break 
                keyCode = cv.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    cv.imwrite("Left_img1.jpg", frame_left)
                    cv.imwrite("Right_img1.jpg", frame_right)
                    break
        finally:
            video_capture_left.release()
            video_capture_right.release()
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
