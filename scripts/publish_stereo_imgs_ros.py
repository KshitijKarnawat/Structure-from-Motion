#!/usr/bin/env python

import numpy as np
import time
import rospy
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os

""" 
gstreamer_pipeline object that returns a GStreamer pipeline for capturing the CSI camera and 
Flip the image by setting the flip_method display_width and display_height determine the size 
of each camera pane in the window on the screen
"""
def gstreamer_pipeline(
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

move = False
cmd = 'S'
left_speed = 0
right_speed = 0
zero_str = 'S' + chr(0) + chr(0)

bridge = CvBridge()
j = 0

try:
    # Initialize ROS node on Jetson nano
    rospy.init_node('jetbot', anonymous=True)
    loop_rate = rospy.Rate(10)

    # Publishers for left image, and right image
    img_pub_l = rospy.Publisher('nano_img_l',Image, queue_size=20)
    img_pub_r = rospy.Publisher('nano_img_r',Image, queue_size=20)
    video_capture_left = cv.VideoCapture(gstreamer_pipeline(sensor_id=0, flip_method=2), cv.CAP_GSTREAMER)
    video_capture_right = cv.VideoCapture(gstreamer_pipeline(sensor_id=1, flip_method=2), cv.CAP_GSTREAMER)

    # Wait for cameras to open
    while not video_capture_left.isOpened() or not video_capture_right.isOpened():
        print("Opening")
        continue


    while not rospy.is_shutdown():
        while j < 20:
            ret_val_left, frame_left = video_capture_left.read()
            ret_val_right, frame_right = video_capture_right.read()
            j += 1
		

        rospy.loginfo('Publishing images')
        img_pub_l.publish(bridge.cv2_to_imgmsg(frame_left))
        img_pub_r.publish(bridge.cv2_to_imgmsg(frame_right))
        loop_rate.sleep()


except KeyboardInterrupt:
    rospy.loginfo("ROS main loop ended")

