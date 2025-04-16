"""
Project 235. Face recognition with deep learning
Description:
Face recognition is the task of identifying or verifying a person based on their facial features. It’s widely used in biometrics, surveillance, secure access systems, and photo tagging. In this project, we’ll use a deep learning-based face recognition model from the face_recognition library, which wraps around dlib's ResNet-based facial embedding extractor.

About:
What It Does:
This project detects and recognizes faces by comparing them to pre-encoded face embeddings. It enables applications like attendance systems, secure door access, tagging people in photos, and multi-face recognition in real-time video. You can scale it up with face databases, real-time webcam feeds, or integrate with DeepFace or InsightFace for more accuracy and speed.
"""
# Install dependencies:
# pip install face_recognition opencv-python
 
import face_recognition
import cv2
import matplotlib.pyplot as plt
 
# Load known image and encode the face
known_image = face_recognition.load_image_file("person1.jpg")  # Replace with a known person's image
known_encoding = face_recognition.face_encodings(known_image)[0]
 
# Load test image and find faces
test_image = face_recognition.load_image_file("group_photo.jpg")  # Replace with test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)
 
# Compare test image faces to known encoding
matches = [face_recognition.compare_faces([known_encoding], face_encoding)[0] for face_encoding in face_encodings]
 
# Convert to BGR for OpenCV display
output_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
 
# Annotate faces
for (top, right, bottom, left), match in zip(face_locations, matches):
    label = "Matched" if match else "Unknown"
    color = (0, 255, 0) if match else (0, 0, 255)
    
    # Draw box and label
    cv2.rectangle(output_image, (left, top), (right, bottom), color, 2)
    cv2.putText(output_image, label, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
# Show result
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Face Recognition Result")
plt.axis('off')
plt.show()