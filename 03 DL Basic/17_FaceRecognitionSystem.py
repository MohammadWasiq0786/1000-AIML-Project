"""
Project 97. Face recognition system
Description:
A face recognition system identifies or verifies a person by analyzing facial features. In this project, we use the face_recognition library (built on dlib) to implement face detection, face encoding, and face comparison, allowing us to recognize faces from images or a webcam feed.

About:
🧠 What This Project Demonstrates:
Uses pre-trained face encodings for recognition

Detects and labels faces in real-time via webcam

Leverages the simplicity and power of face_recognition library
"""

# Install if not already: pip install face_recognition opencv-python
 
import face_recognition
import cv2
import numpy as np
import os
 
# Load known faces (assume folder "known_faces/name.jpg")
known_encodings = []
known_names = []
 
known_faces_dir = 'known_faces'  # Replace with your folder path
 
for file in os.listdir(known_faces_dir):
    img_path = os.path.join(known_faces_dir, file)
    img = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(img)
    if encoding:
        known_encodings.append(encoding[0])
        known_names.append(os.path.splitext(file)[0])
 
# Load test image or use webcam
cap = cv2.VideoCapture(0)  # Use webcam; alternatively load image with cv2.imread()
 
print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = small_frame[:, :, ::-1]
 
    # Detect faces and encode
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
 
    # Compare with known encodings
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"
 
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
 
        # Scale back up face location
        top, right, bottom, left = top*4, right*4, bottom*4, left*4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
 
    # Display result
    cv2.imshow("Face Recognition", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()