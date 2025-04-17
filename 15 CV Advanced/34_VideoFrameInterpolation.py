"""
Project 594: Video Frame Interpolation
Description:
Video frame interpolation involves generating intermediate frames between two consecutive frames in a video to create smooth slow-motion effects or improve video frame rates. This process requires understanding temporal continuity and motion patterns in the video. In this project, we will implement frame interpolation using deep learning techniques, such as Deep Voxel Flow or FlowNet.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Load two consecutive frames from a video for interpolation
frame1 = cv2.imread("path_to_frame1.jpg")  # Replace with actual image paths
frame2 = cv2.imread("path_to_frame2.jpg")
 
# 2. Convert the frames to grayscale (required for optical flow computation)
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
 
# 3. Compute optical flow between the two frames using Farneback method
flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
 
# 4. Generate an intermediate frame (e.g., by using linear interpolation based on the flow)
height, width = flow.shape[:2]
intermediate_frame = np.zeros_like(frame1)
 
for y in range(height):
    for x in range(width):
        dx, dy = flow[y, x]
        intermediate_frame[y, x] = frame1[y + int(dy), x + int(dx)]  # Simple interpolation
 
# 5. Visualize the original and interpolated frames
plt.figure(figsize=(10, 5))
 
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
plt.title("Original Frame 1")
 
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(intermediate_frame, cv2.COLOR_BGR2RGB))
plt.title("Interpolated Frame")
 
plt.show()