"""
Project 593: Multi-view 3D Reconstruction
Description:
Multi-view 3D reconstruction involves creating a 3D model from multiple 2D images taken from different viewpoints. This process is important for applications like computer vision, augmented reality, and robotics. In this project, we will use multiple 2D images and perform 3D reconstruction by estimating the 3D points corresponding to objects in the images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Load multiple images from different viewpoints (e.g., images taken from different angles)
image1 = cv2.imread("path_to_image1.jpg", cv2.IMREAD_GRAYSCALE)  # Replace with actual image paths
image2 = cv2.imread("path_to_image2.jpg", cv2.IMREAD_GRAYSCALE)
 
# 2. Detect keypoints and compute descriptors using SIFT or ORB
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)
 
# 3. Match the keypoints between the two images using a brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
 
# 4. Extract matched keypoints' coordinates
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
 
# 5. Find the essential matrix using the matched points
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
 
# 6. Recover the camera pose (relative position and orientation) between the two views
_, H = cv2.findHomography(pts1, pts2, cv2.RANSAC)
 
# 7. Triangulate the matched points to get 3D coordinates
# Assume that the camera matrices are known (can be computed using camera calibration)
P1 = np.eye(3, 4)  # Camera projection matrix for the first image
P2 = np.hstack([H[:3, :3], np.zeros((3, 1))])  # Camera projection matrix for the second image
 
# Triangulate points to obtain 3D coordinates
points_3d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
 
# 8. Convert homogeneous coordinates to 3D space
points_3d /= points_3d[3]
 
# 9. Visualize the 3D reconstruction (simplified, here we plot in 3D space)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[0], points_3d[1], points_3d[2], c=points_3d[2], cmap='viridis')
ax.set_title("3D Reconstruction from Multiple Views")
plt.show()