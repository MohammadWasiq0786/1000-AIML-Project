"""
Project 599: Video Stabilization
Description:
Video stabilization is the process of removing unwanted camera shake or jitter from a video, making the footage smoother. This task is important for improving the viewing experience in handheld videos or videos shot in shaky conditions. In this project, we will implement video stabilization using methods such as motion estimation and frame alignment.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Load the video (use a video file path or URL)
video_path = "path_to_video.mp4"  # Replace with an actual video path
cap = cv2.VideoCapture(video_path)
 
# 2. Get the first frame and initialize the transformation matrix (identity matrix)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
transform_matrix = np.eye(3, 3, dtype=np.float32)
 
# 3. Prepare for video stabilization (store the transformed frames)
stabilized_frames = []
 
# 4. Process each frame for stabilization
while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break
 
    # 5. Convert the current frame to grayscale for optical flow computation
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
 
    # 6. Find optical flow between the previous and current frames
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
 
    # 7. Calculate the translation vector
    dx, dy = np.mean(flow, axis=(0, 1))  # Average displacement of flow
 
    # 8. Create a transformation matrix to stabilize the video
    translation_matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
 
    # 9. Apply the transformation to the current frame
    stabilized_frame = cv2.warpAffine(curr_frame, translation_matrix, (curr_frame.shape[1], curr_frame.shape[0]))
 
    # 10. Add the stabilized frame to the list
    stabilized_frames.append(stabilized_frame)
 
    # 11. Update the previous frame and grayscale image
    prev_frame = curr_frame
    prev_gray = curr_gray
 
# 12. Release the video capture object
cap.release()
 
# 13. Display the original and stabilized video
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(prev_frame)
axs[0].set_title("Original Frame")
axs[0].axis('off')
 
axs[1].imshow(stabilized_frames[-1])  # Display the last stabilized frame
axs[1].set_title("Stabilized Frame")
axs[1].axis('off')
 
plt.show()
 
# Optionally, save the stabilized video
output_video_path = "stabilized_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify codec
out = cv2.VideoWriter(output_video_path, fourcc, 30, (prev_frame.shape[1], prev_frame.shape[0]))
 
for frame in stabilized_frames:
    out.write(frame)
 
out.release()

"""
This code demonstrates video stabilization by estimating the optical flow between frames and using the translation components of the flow to stabilize the video.
"""