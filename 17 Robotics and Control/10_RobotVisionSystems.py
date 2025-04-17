"""
Project 650: Robot Vision Systems
Description:
Robot vision systems enable robots to perceive and understand their environment using cameras and sensors. These systems are essential for tasks such as object detection, scene recognition, and navigation. In this project, we will implement a simple robot vision system using OpenCV to detect and track objects in real-time. The system will include steps like image processing, feature detection, and object tracking.
"""

import cv2
import numpy as np
 
# 1. Load the video or camera feed
cap = cv2.VideoCapture(0)  # Use webcam as input (or replace with a video file path)
 
# 2. Set up the object detection (using ORB feature detector for simplicity)
orb = cv2.ORB_create()
 
# 3. Initialize the tracking variables
previous_keypoints = None
previous_descriptors = None
 
# 4. Start the video capture loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale (required for ORB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Detect ORB keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray_frame, None)
 
    # If it's the first frame, store the keypoints and descriptors
    if previous_keypoints is None:
        previous_keypoints = keypoints
        previous_descriptors = descriptors
        continue
 
    # 5. Match features between the current frame and the previous frame
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(previous_descriptors, descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
 
    # 6. Draw matches between the frames
    result_frame = cv2.drawMatches(frame, previous_keypoints, frame, keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
    # Display the result
    cv2.imshow("Robot Vision System - Object Detection and Tracking", result_frame)
 
    # Update the previous frame keypoints and descriptors
    previous_keypoints = keypoints
    previous_descriptors = descriptors
 
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 7. Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()