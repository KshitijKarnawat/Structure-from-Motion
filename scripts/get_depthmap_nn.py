#!/usr/bin/env python

import numpy as np
import time
import serial
import rospy
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os

# Initializing serial port to communicate with arduino nano
ser = serial.Serial(port="/dev/ttyUSB0", baudrate=115200, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
# display = jetson.utils.videoOutput("display://0")

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

# Function to make command in a format as per arduino code
def make_command(cmd,left_speed,right_speed):
    cmd_to_arduino = cmd + chr(int(left_speed)) + chr(int(right_speed))
    return cmd_to_arduino

def nothing(x):
    pass

# Function to rotate a robot for time t
def rotate_robot(t):
    rmc = 0
    n_rmc = 10
    data_to_arduino = make_command('S',109,46)
    while rmc < n_rmc:
        ser.write(data_to_arduino.encode())
        time.sleep(t/n_rmc)
        rmc += 1

    rmc = 0
    print("stop")
    data_to_arduino = make_command('S',0,0)
    while rmc < n_rmc:
        ser.write(data_to_arduino.encode())
        time.sleep(0.4/n_rmc)
        rmc += 1

# Read the model of DPT using cv2 dnn module
model_path = "models/"
model_name = "model-small.onnx"
model = cv.dnn.readNet(model_path + model_name)

if model.empty():
    print("Model not loaded")

# Arduino communication initialization variables
move = False
cmd = 'S'
left_speed = 0
right_speed = 0
zero_str = 'S' + chr(0) + chr(0)

# Initialize CV bridge
bridge = CvBridge()
j = 0
rc = 10
im_ctr = 0



try:
    # Code to find depth map using neural networks
    # Initialize ROS node on Jetson nano
    rospy.init_node('jetbot', anonymous=True)
    loop_rate = rospy.Rate(10)

    # Publishers for left image, right image, and the depth map
    img_pub_l = rospy.Publisher('nano_img_l',Image, queue_size=1)
    img_pub_r = rospy.Publisher('nano_img_r',Image, queue_size=1)
    depth_mono = rospy.Publisher('nano_img_depth',Image, queue_size=1)

    # Create camera capture objects for left and right cameras
    video_capture_left = cv.VideoCapture(gstreamer_pipeline(sensor_id=0, flip_method=2), cv.CAP_GSTREAMER)
    video_capture_right = cv.VideoCapture(gstreamer_pipeline(sensor_id=1, flip_method=2), cv.CAP_GSTREAMER)
    #video_capture_left.set(cv.CAP_PROP_BUFFERSIZE, 1)
    #video_capture_right.set(cv.CAP_PROP_BUFFERSIZE, 1)

    # Wait for cameras to open
    while not video_capture_left.isOpened() or not video_capture_right.isOpened():
        print("Opening")
        continue


    while not rospy.is_shutdown():
        # Skip 20 frames to get stable frames for later processing
        while j < 20:
            ret_val_left, frame_left = video_capture_left.read()
            ret_val_right, frame_right = video_capture_right.read()
            j += 1
        

        rospy.loginfo('Publishing images')
        if rc > 0:
            h, w, c = frame_right.shape
            blob = cv.dnn.blobFromImage(frame_right, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)

            # Set input to the model
            model.setInput(blob)

            output = model.forward()
            output = output[0,:,:]
            output = cv.resize(output, (w,h))
            output = cv.normalize(output, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype= cv.CV_32F)

            cv.imwrite('depth.png',output)

            img_pub_l.publish(bridge.cv2_to_imgmsg(frame_left))#, encoding="bgr8"))
            img_pub_r.publish(bridge.cv2_to_imgmsg(frame_right))#, encoding="bgr8"))
            depth_mono.publish(bridge.cv2_to_imgmsg(output))
            rotate_robot(0.7)
            time.sleep(1)
            cv.imwrite("newl_"+str(im_ctr)+".jpg", frame_left)
            cv.imwrite("newr_"+str(im_ctr)+".jpg", frame_right)
            im_ctr+=1
        
        rc -= 1
        # Release cams at each iteration as the images are not updated if this is not done
        video_capture_left.release()
        video_capture_right.release()
        j = 0

        time.sleep(0.5)

        # Again starting the cameras for next frame capturing
        video_capture_left = cv.VideoCapture(gstreamer_pipeline(sensor_id=0, flip_method=2), cv.CAP_GSTREAMER)
        video_capture_right = cv.VideoCapture(gstreamer_pipeline(sensor_id=1, flip_method=2), cv.CAP_GSTREAMER)

        # wait for cameras to open for next time
        while not video_capture_left.isOpened() or not video_capture_right.isOpened():
            print("Opening")
            continue

        # Only capture until rc (record counter) reduces to zero
        if rc <= 0:
            rospy.loginfo("Done capturing")
            break

        loop_rate.sleep()


except KeyboardInterrupt:
    rospy.loginfo("ROS main loop ended")

