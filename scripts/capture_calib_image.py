# Code to capture stereo images and store with increasing index to use later

import cv2
import time

""" 
gstreamer_pipeline object that returns a GStreamer pipeline for capturing the CSI camera and 
Flip the image by setting the flip_method display_width and display_height determine the size 
of each camera pane in the window on the screen
"""

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


def show_camera():
    window_title_left = "CSI Camera left"
    window_title_right = "CSI Camera right"
    i = 0

    # To flip the image, modify the flip_method parameter (2 to flip vertically)
    print(gstreamer_pipeline_left(flip_method=0))
    try:
        while True:
            key = input("Enter f to capture image: ")
            if str(key) == 'f':
                j = 0
                # Create camera capture objects for left and right cameras
                video_capture_left = cv2.VideoCapture(gstreamer_pipeline_left(flip_method=2), cv2.CAP_GSTREAMER)
                video_capture_right = cv2.VideoCapture(gstreamer_pipeline_right(flip_method=2), cv2.CAP_GSTREAMER)

                # If cameras are open
                if video_capture_left.isOpened() and video_capture_right.isOpened():
                    print("Capturing image")
                    # Skip 15 frames to get stable frames for later stages
                    while j < 15:
                        ret_val_left, frame_left = video_capture_left.read()
                        ret_val_right, frame_right = video_capture_right.read()
                        j += 1
                    
                    frame_left = cv2.flip(frame_left, 1)                
                    frame_right = cv2.flip(frame_right, 1)

                    cv2.imwrite("Left_img_" + str(i) + ".jpg", frame_left)
                    cv2.imwrite("Right_img_" + str(i) + ".jpg", frame_right)
                    video_capture_left.release()
                    video_capture_right.release()
                    i += 1
 
                time.sleep(0.01)

    finally:
        video_capture_left.release()
        video_capture_right.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    show_camera()
