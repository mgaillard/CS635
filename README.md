# CS635: Assignments

A repo that contains my assignments for the class CS-635 at Purdue University.

## Assignment 1: Camera Calibration
The objective of this assignment is to calibrate the intrinsinc and extrinsic parameters of a camera. I used my phone to capture two images from a chessboard pattern and Implemented Zhang's Camera Calibration algorithm. The verification of proper calibration is by visual re-projection of the calibration points. I also compare my results to the OpenCV [cv::calibrateCamera](https://docs.opencv.org/3.4.1/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d) function.

Reference paper: Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE Transactions on pattern analysis and machine intelligence*, 22(11), 1330-1334.

## Author
Mathieu Gaillard
