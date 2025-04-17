"""
Project 648: Visual Odometry Implementation
Description:
Visual odometry (VO) is the process of estimating the motion of a camera (or robot) by analyzing the sequence of images it captures. This technique helps determine the position and orientation of the robot over time without relying on external sensors like GPS. In this project, we will implement a simple visual odometry system using feature matching between consecutive images to estimate the robot's motion. We will use techniques such as Feature Detection and Optical Flow to track motion.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Load two consecutive images (simulating two frames of a video)
image1 = cv2.imread('frame1.png', cv2.IMREAD_GRAYSCALE)  # Replace with actual image paths
image2 = cv2.imread('frame2.png', cv2.IMREAD_GRAYSCALE)  # Replace with actual image paths
 
# 2. Feature detection (using ORB - Oriented FAST and Rotated BRIEF)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)
 
# 3. Match features using Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
 
# 4. Sort the matches based on distance (best matches come first)
matches = sorted(matches, key = lambda x:x.distance)
 
# 5. Visualize the feature matches between the two images
img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(10, 10))
plt.imshow(img_matches)
plt.title("Feature Matches Between Consecutive Frames")
plt.show()
 
# 6. Compute Essential Matrix (using RANSAC to remove outliers)
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1.0)
 
# 7. Recover the camera motion (rotation and translation) from the Essential matrix
_, R, t, mask = cv2.recoverPose(E, pts1, pts2)
 
# 8. Show the estimated camera motion (rotation matrix and translation vector)
print("Estimated Rotation Matrix:\n", R)
print("Estimated Translation Vector:\n", t)
 
# 9. Visualize the camera motion (optional, e.g., plotting camera trajectory)
# Assuming we can track the camera's trajectory using the translation vector
 
# Initialize a trajectory plot
trajectory = np.zeros((100, 3))  # Placeholder for camera's estimated position
trajectory[0] = np.array([0, 0, 0])  # Start at origin
 
# Simulate a simple 2D trajectory (this is just a basic example)
for i in range(1, 100):
    trajectory[i] = trajectory[i - 1] + t.flatten()  # Update position based on translation
 
# Plot the trajectory
plt.figure(figsize=(8, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Estimated Camera Path")
plt.scatter(trajectory[:, 0], trajectory[:, 1], color='red', s=2)
plt.title("Estimated Camera Path using Visual Odometry")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.show()