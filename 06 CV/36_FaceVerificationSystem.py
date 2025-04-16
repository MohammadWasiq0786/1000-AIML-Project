"""
Project 236. Face verification system
Description:
Face verification answers the question: “Are these two faces from the same person?” It’s a key component in biometric authentication systems, such as phone unlock, secure access gates, and identity validation platforms. This project implements face verification by computing the embedding distance between two face images using deep learning.

About:
What It Does:
This project compares two facial images and determines if they belong to the same identity, based on how similar their deep feature encodings are. It’s the foundation for login with face, fraud detection, and digital identity verification systems. You can expand it using Siamese networks, DeepFace, or ArcFace for higher robustness and accuracy.
"""
# Install dependencies:
# pip install face_recognition opencv-python matplotlib
 
import face_recognition
import matplotlib.pyplot as plt
from PIL import Image
 
# Load two face images: one reference, one to verify
image1 = face_recognition.load_image_file("person_A.jpg")  # Reference image
image2 = face_recognition.load_image_file("person_B.jpg")  # Image to verify
 
# Get face encodings (128-d vector per face)
encoding1 = face_recognition.face_encodings(image1)
encoding2 = face_recognition.face_encodings(image2)
 
# Ensure both images contain faces
if not encoding1 or not encoding2:
    print("❌ Face not found in one or both images.")
    exit()
 
# Compute Euclidean distance between encodings
distance = face_recognition.face_distance([encoding1[0]], encoding2[0])[0]
 
# Set a threshold for verification (commonly around 0.6)
threshold = 0.6
is_same_person = distance < threshold
 
# Display results
print(f"\n🔍 Face Verification Result:")
print(f"Distance between faces: {distance:.4f}")
print("✅ Match (Same Person)" if is_same_person else "❌ No Match (Different People)")
 
# Show both images side by side
plt.figure(figsize=(8, 4))
 
plt.subplot(1, 2, 1)
plt.imshow(Image.open("person_A.jpg"))
plt.title("Person A")
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(Image.open("person_B.jpg"))
plt.title("Person B")
plt.axis('off')
 
plt.suptitle("Face Verification", fontsize=14)
plt.tight_layout()
plt.show()