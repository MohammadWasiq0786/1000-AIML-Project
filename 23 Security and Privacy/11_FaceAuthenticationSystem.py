"""
Project 891. Face Authentication System

A face authentication system verifies a user by comparing a live facial image with a stored reference image. In this project, we’ll use the popular face_recognition library to perform face encoding comparison between a stored face and a new input.

Key Points:
Face encodings are 128-dimensional feature vectors.

compare_faces returns a boolean match, while face_distance gives the similarity score.

Can be extended with live capture via webcam (using OpenCV) and liveness detection (to prevent spoofing).
"""

import face_recognition
 
# Load reference image (enrolled user)
reference_image = face_recognition.load_image_file("user.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]
 
# Load login image (input to verify)
login_image = face_recognition.load_image_file("login.jpg")
login_encoding = face_recognition.face_encodings(login_image)[0]
 
# Compare face encodings
matches = face_recognition.compare_faces([reference_encoding], login_encoding)
distance = face_recognition.face_distance([reference_encoding], login_encoding)[0]
 
# Define threshold (face_recognition default is ~0.6)
threshold = 0.6
print(f"Face Distance: {distance:.4f}")
 
if matches[0] and distance < threshold:
    print("✅ Face Authentication Successful: Access Granted.")
else:
    print("❌ Face Authentication Failed: Access Denied.")