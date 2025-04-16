"""
Project 147. Pose estimation system
Description:
Pose estimation is the task of detecting keypoints (like elbows, knees, or shoulders) on the human body in images or video. This project implements a pose estimation system using a pretrained deep learning model (e.g., OpenPose, MoveNet, or BlazePose) to identify and track human body landmarks from a webcam or image input.

About:
🧠 What This Project Demonstrates:
Uses MediaPipe’s Pose solution for fast and accurate landmark detection

Estimates 33 keypoints on the body (e.g., head, shoulders, knees, feet)

Runs in real-time from webcam input
"""

# Install if not already: pip install mediapipe opencv-python
 
import cv2
import mediapipe as mp
 
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
 
pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
 
# Capture video from webcam
cap = cv2.VideoCapture(0)
 
print("📷 Starting real-time pose estimation... Press 'q' to quit.")
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    # Convert BGR to RGB for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose_estimator.process(image)
 
    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    # Draw landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
 
    cv2.imshow('Pose Estimation', image)
 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()