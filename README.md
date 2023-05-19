# Structure-from-Motion
Final Project for ENPM673 ROS package and Jetson codes

## Overview

Images were captured using 2 Raspberry pi cameras setup in custom 3D printed mounts, the interface used is CSI through CSI camera headeers. Also contains 

## Team Members

Kshitij Karnawat (UID: kshitij 119188651)
Hritvik Choudhary (UID: hac 119208793)
Abhimanyu Saxena (UID: asaxena4 119342763)
Mudit Singal (UID: msingal 119262689)
Raajith Gadam (UID: raajithg 119461167) 

## Dependencies

## How to Setup Repo in Local Workspace
**Works with ROS melodic**
You have to use ROS melodic and python 2.7 for majority of the codes in this repo.
1. Download the folder sfm from the git repo or git clone in local machine
2. Copy and paste the folder sfm into your catkin workspace (~/catkin_ws/src)
3. cd ~/catkin_ws
4. catkin_make
5. cd ~/catkin_ws/src/sfm/scripts
6. chmod +x *.py
7. ./rotate_and_pub.py to run the rotate and publish stereo images
8. ./publish_stereo_imgs_ros.py to run code to publish stereo images captured by jetson nano
9. ./get_depthmap_nn.py to run the code to rotate the robot, capture stereo images, compute the depth map using DPT model from right images and publish as separate topics on the ROS network



### Standalone codes
The python files: _capture_calib_image.py_ and _get_depth_map.py_ are standalone python files that can be run on the jetson nano to capture images from a stereo setup and get depth map from built-in opencv functions.
Jetson nano standard image already ships with numpy and opencv. You can install matplotlib and pyserial(if required) using the following commands:
```sh
sudp apt-get install python-pip
sudo apt-get install python3-pip
pip install pyserial
pip3 install pyserial
```


#### Instructions to run _capture_calib_image.py_:
```sh
python capture_calib_image.py
press f to capture images and save to the same folder
```

#### Instructions to run _get_depth_map.py_:
```sh
python3 get_depth_map.py
or 
python get_depth_map.py
```

#### Arduino code:
1. Make connections between arduino nano and motor driver as per the pinout given in ino file
2. Flash the Arduino with the hbridge_script.ino
3. Connect the Arduino nano and Jetson nano with a usb cable for arduino


### Run the following at jetson nano side if you encounter error while accessing serial port
```sh
sudo chmod -R 777 /dev/ttyUSB0
```

