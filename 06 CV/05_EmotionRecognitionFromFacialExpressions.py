"""
Project 205. Emotion recognition from facial expressions
Description:
Emotion Recognition from facial expressions classifies human emotions like happy, sad, angry, or surprised based on facial cues. It has real-world uses in education, healthcare, marketing, and AI companions. In this project, we use a pretrained CNN model trained on the FER-2013 dataset to detect emotions in face images.


About:
🧠 What This Project Demonstrates:
Detects faces and classifies them into common emotions

Uses a pretrained model via FER library (based on CNN + FER-2013 dataset)

Returns probabilities for multiple emotions per face
"""
# Install if not already: pip install fer opencv-python
 
from fer import FER
import cv2
import matplotlib.pyplot as plt
 
# Load image with a face (or use webcam capture)
image_path = 'emotion_face.jpg'
image = cv2.imread(image_path)
 
# Convert to RGB for FER
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# Initialize FER detector
detector = FER(mtcnn=True)  # MTCNN helps with better face detection
 
# Detect emotions
result = detector.detect_emotions(rgb_img)
 
# Display results
if result:
    emotions = result[0]["emotions"]
    top_emotion = max(emotions, key=emotions.get)
    print(f"🧠 Detected Emotion: {top_emotion} (Confidence: {emotions[top_emotion]:.2f})")
else:
    print("😕 No face detected in the image.")
 
# Show image
plt.imshow(rgb_img)
plt.title(f"Detected Emotion: {top_emotion}")
plt.axis('off')
plt.show()