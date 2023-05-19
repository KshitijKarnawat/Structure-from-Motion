# Structure-from-Motion
Final Project for ENPM673

## Overview

To construct a 3D point cloud of the surroundings using a mobile ground robot (Map Generation). The robot scans it's surroundings by collecting images from different view points and processes it to reconstruct the space in 3D using SfM. 

## Abstract

Structure from motion is a low-cost alternative to construct 3D representation of an object. With the increase in the cost of 3D ranging sensors like LIDAR, laser scanners etc. It makes sense to use photogrammetry techniques to map a space in 3D using 2D images taken from different viewpoints. In the scenario of indoor mapping, due to complexity and closeness of the space, using SfM and minimizing the reprojection error of the matching points we can effectively perform 3D reconstruction using fewer images. The SfM pipeline starts with capturing multiple images of the indoor space from varying viewpoints, followed by feature extraction, and matching, geometric verification and refining construction. This serves as the foundation for our model reconstruction. The scene points are triangulated, and outliers are filtered using RANSAC to give a robust estimation of the space in 3D. To fulfill the requirements of the project, the data is collected using a high-definition camera (Intel Realsense) mounted on a mobile ground robot and processing is done onboard using Jetson Nano. This technique can be used to scan and reconstruct land terrains, warehouses and in autonomous vehicles to map its environment.

## Team Members

Kshitij Karnawat (UID: kshitij 119188651)
Hritvik Choudhary (UID: hac 119208793)
Abhimanyu Saxena (UID: asaxena4 119342763)
Mudit Singal (UID: msingal 119262689)
Raajith Gadam (UID: raajithg 119461167) 

## Dependencies

## How to Setup Repo in Local Workspace

### Run the following if you encounter error while accessing serial port
```sh
sudo chmod -R 777 /dev/ttyUSB0
```

```sh
mkdir Structure-from-Motion
cd Structure-from-Motion
git init
git remote add origin https://github.com/KshitijKarnawat/Structure-from-Motion
git switch devel
git pull
```
