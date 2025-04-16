"""
Project 239. Eye tracking implementation
Description:
Eye tracking involves detecting and following the movement and position of a person's eyes, particularly the pupils or gaze direction. It’s used in attention analysis, accessibility tools, gaming, UX research, and driver monitoring systems. In this project, we’ll implement a basic eye tracking system using dlib for face and landmark detection and OpenCV to highlight pupil movement.

About:
What It Does:
This project tracks the center of both eyes (pupils) in real-time, enabling basic eye movement analysis. You can extend this into a gaze direction estimator, use it for blink detection, or integrate it with accessibility software and human-computer interaction systems for hands-free control or attention tracking.
"""
# Install required packages:
# pip install opencv-python dlib imutils
 
import cv2
import dlib
import numpy as np
 
# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download this model from dlib's model zoo
predictor = dlib.shape_predictor(predictor_path)
 
# Define indexes for left and right eyes in the 68-point model
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
 
# Function to get eye region and pupil center
def get_eye_center(landmarks, eye_points):
    eye_region = np.array([(landmarks[i][0], landmarks[i][1]) for i in eye_points], np.int32)
    x, y, w, h = cv2.boundingRect(eye_region)
    roi = gray[y:y+h, x:x+w]
    roi = cv2.equalizeHist(roi)
    _, threshold = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY_INV)
 
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assume largest contour is the pupil
        contour = max(contours, key=cv2.contourArea)
        (cx, cy), _ = cv2.minEnclosingCircle(contour)
        return int(x + cx), int(y + cy)
    return None
 
# Start webcam feed
cap = cv2.VideoCapture(0)
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
 
    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
 
        # Track both eyes
        left_center = get_eye_center(landmarks, LEFT_EYE)
        right_center = get_eye_center(landmarks, RIGHT_EYE)
 
        # Draw circles on pupil centers
        if left_center:
            cv2.circle(frame, left_center, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"L Eye: {left_center}", (left_center[0] + 10, left_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
 
        if right_center:
            cv2.circle(frame, right_center, 5, (255, 0, 0), -1)
            cv2.putText(frame, f"R Eye: {right_center}", (right_center[0] + 10, right_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
 
    cv2.imshow("Eye Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break
 
cap.release()
cv2.destroyAllWindows()