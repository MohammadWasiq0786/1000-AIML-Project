"""
Project 240. Gaze estimation system
Description:
Gaze estimation aims to determine where a person is looking, often using just webcam input. It’s essential in attention tracking, eye-controlled interfaces, driver alert systems, and psychological studies. In this project, we’ll implement a basic gaze estimator using dlib facial landmarks, focusing on eye direction (left, center, right) by comparing the pupil's position within the eye region.

About:
What It Does:
This project estimates whether the person is looking left, center, or right by analyzing pupil position relative to the eye. It’s a simplified yet effective approach for screen interaction, attention tracking, or focus-based automation. You can upgrade it with CNN-based models like GazeNet, RT-GENE, or use 3D head-pose estimation for precise gaze direction.
"""
# Install dependencies:
# pip install opencv-python dlib imutils
 
import cv2
import dlib
import numpy as np
 
# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download from dlib's model zoo
predictor = dlib.shape_predictor(predictor_path)
 
# Eye landmark indices from 68-point model
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
 
# Calculate gaze direction by analyzing pupil location within eye
def get_gaze_direction(eye_points, gray_img):
    eye_region = np.array(eye_points, dtype=np.int32)
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
 
    eye_img = gray_img[min_y:max_y, min_x:max_x]
    _, threshold = cv2.threshold(eye_img, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        (cx, _), _ = cv2.minEnclosingCircle(max_cnt)
        pos_ratio = cx / (max_x - min_x)
        if pos_ratio <= 0.35:
            return "Looking Left"
        elif pos_ratio >= 0.65:
            return "Looking Right"
        else:
            return "Looking Center"
    return "Undetected"
 
# Start webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
 
    for face in faces:
        landmarks = predictor(gray, face)
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
 
        # Get gaze direction for each eye
        left_eye_pts = [points[i] for i in LEFT_EYE]
        right_eye_pts = [points[i] for i in RIGHT_EYE]
 
        left_gaze = get_gaze_direction(left_eye_pts, gray)
        right_gaze = get_gaze_direction(right_eye_pts, gray)
 
        # Display gaze direction
        gaze = left_gaze if left_gaze == right_gaze else "Uncertain"
        cv2.putText(frame, f"Gaze: {gaze}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
        # Draw eye regions
        cv2.polylines(frame, [np.array(left_eye_pts)], True, (255, 0, 0), 1)
        cv2.polylines(frame, [np.array(right_eye_pts)], True, (255, 0, 0), 1)
 
    cv2.imshow("Gaze Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break
 
cap.release()
cv2.destroyAllWindows()