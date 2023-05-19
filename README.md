# Structure-from-Motion

Final Project for ENPM673

## Overview

To construct a 3D point cloud of the surroundings using a mobile ground robot (Map Generation). The robot scans it's surroundings by collecting images from different view points and processes it to reconstruct the space in 3D using SfM.

## Abstract

Structure from motion is a low-cost alternative to construct 3D representation of an object. With the increase in the cost of 3D ranging sensors like LIDAR, laser scanners etc. It makes sense to use photogrammetry techniques to map a space in 3D using 2D images taken from different viewpoints. In the scenario of indoor mapping, due to complexity and closeness of the space, using SfM and minimizing the reprojection error of the matching points we can effectively perform 3D reconstruction using fewer images. The SfM pipeline starts with capturing multiple images of the indoor space from varying viewpoints, followed by feature extraction, and matching, geometric verification and refining construction. This serves as the foundation for our model reconstruction. The scene points are triangulated, and outliers are filtered using RANSAC to give a robust estimation of the space in 3D. To fulfill the requirements of the project, the data is collected using a high-definition camera (Intel Realsense) mounted on a mobile ground robot and processing is done onboard using Jetson Nano. This technique can be used to scan and reconstruct land terrains, warehouses and in autonomous vehicles to map its environment.

## Team Members

1. Kshitij Karnawat (<kshitij@terpmail.edu>)
2. Hritvik Choudhary (<hac@umd.edu>)
3. Abhimanyu Saxena (<asaxena4@umd.edu>)
4. Mudit Singal (<msingal@umd.edu>)
5. Raajith Gadam (<raajithg@umd.edu>)

## Dependencies

- numpy
- matplotlib
- opencv-python
- opencv-contrib-python
- open3d
- glob
- tqdm
- glob
- os
- errno
- torch
- math
- einops
- timm
- wandb
- PIL
- random
- json
- sklearn

## Running

Calibrate the camera using the `calibrate.py` script. Store all the calibration images in  `Data/Calibration images` folder.

```sh
cd scripts
python3 calibrate.py
```

You can save all the images in the `Data` folder instead if hardware is not available.

To get the depth estimation using the stereo method run the following lines.

```sh
cd scripts
python3 generate_point_cloud.py
```

To get the depth estimation using the DPT run the following lines.

```sh
cd scripts
python3 depth_nn.py
```

To get the stitched point cloud run the following lines.

```sh
cd scripts
python3 generate_point_cloud.py
```
