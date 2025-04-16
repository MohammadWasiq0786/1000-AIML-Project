"""
Project 221. Human pose estimation
Description:
Human Pose Estimation involves detecting key points of the human body (e.g., joints like elbows, wrists, knees) from images or video. It’s used in fitness tracking, motion analysis, gaming, gesture recognition, and more. In this project, we’ll use MediaPipe by Google, which offers a lightweight and fast pose estimation solution.

About:
What It Does:
This project detects body joints and outlines the human skeleton. It can be used for real-time fitness feedback, sports performance analysis, rehabilitation, and interactive systems like gesture-based gaming or smart mirrors. You can extend it with pose classification (e.g., yoga pose detection) or multi-person tracking.
"""
# Install dependencies:
# pip install mediapipe opencv-python matplotlib
 
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
 
# Initialize MediaPipe pose estimator
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils
 
# Load an image with a person
image_path = "person_pose.jpg"  # Replace with your image
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# Process the image to detect human pose landmarks
results = pose.process(rgb_image)
 
# Draw the landmarks if found
if results.pose_landmarks:
    annotated_image = rgb_image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,  # Draw connections between keypoints
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )
else:
    annotated_image = rgb_image.copy()
    print("No human pose detected.")
 
# Display the result
plt.figure(figsize=(10, 6))
plt.imshow(annotated_image)
plt.title("Human Pose Estimation")
plt.axis('off')
plt.show()