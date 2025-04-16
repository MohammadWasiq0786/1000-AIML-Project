"""
Project 203. Face detection implementation
Description:
Face Detection identifies and locates human faces within images or video frames. It's foundational for face recognition, emotion analysis, and camera-based user interaction. In this project, we use OpenCV’s Haar Cascades to perform fast and reliable face detection in images and real-time webcam feeds.

About:
🧠 What This Project Demonstrates:
Detects human faces using Haar Cascade Classifiers

Works on static images and can be adapted for real-time webcam input

Efficient and lightweight — ideal for entry-level face detection systems
"""

# Install if not already: pip install opencv-python
 
import cv2
 
# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
# Load an image with faces (replace with your own image path)
image_path = 'faces.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Detect faces (returns a list of rectangles)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
 
# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
# Show the result
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()