"""
Project 204. Facial landmark detection
Description:
Facial Landmark Detection identifies key points on a face (like eyes, nose, mouth, chin). It’s crucial for applications like face alignment, emotion recognition, AR filters, and 3D face modeling. In this project, we use Dlib and OpenCV to detect 68 facial landmarks from live webcam or image input.

About:
🧠 What This Project Demonstrates:
Detects 68 facial landmark points (eyes, eyebrows, nose, lips, jaw)

Uses Dlib’s shape predictor for accurate point mapping

Works on both images and live video feeds
"""

# Install dependencies:
# pip install opencv-python dlib
 
import cv2
import dlib
 
# Load pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this model file from: http://dlib.net/files/
 
# Load an image (or use webcam for live detection)
image_path = 'face_sample.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Detect faces
faces = detector(gray)
 
# Draw landmarks on each detected face
for face in faces:
    landmarks = predictor(gray, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
 
# Show result
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()