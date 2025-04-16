"""
Project 115. Emotion recognition from images
Description:
Emotion recognition from images involves detecting human facial expressions and classifying them into emotions such as happy, sad, angry, surprised, etc. In this project, we use a CNN-based model trained on the FER-2013 dataset to recognize emotions from facial images using OpenCV and Keras.

About:
😄😢😠 What This Project Demonstrates:
Detects faces using Haar cascades

Classifies facial expressions into basic emotions using a CNN

Can run on live webcam or static images
"""

# Install if not already: pip install opencv-python tensorflow keras numpy
 
import cv2
import numpy as np
from tensorflow.keras.models import load_model
 
# Load pre-trained emotion recognition model (use any trained CNN or download one from GitHub)
model = load_model("emotion_model.h5")  # Replace with the path to your model
 
# Emotion labels (adjust based on your model)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
 
# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
 
# Load image or use webcam
cap = cv2.VideoCapture(0)  # Replace with image path for static image testing
 
print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))  # FER-2013 uses 48x48
        face_normalized = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_normalized, axis=(0, -1))  # Shape: (1, 48, 48, 1)
 
        prediction = model.predict(face_input)
        emotion = emotion_labels[np.argmax(prediction)]
 
        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
 
    cv2.imshow("Emotion Recognition", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()