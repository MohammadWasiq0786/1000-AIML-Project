"""
Project 893. Iris Recognition Implementation

Iris recognition authenticates users based on the unique patterns in their irises. It is one of the most accurate biometric methods. In this project, we simulate iris recognition by using image preprocessing and feature extraction with OpenCV's contour and histogram analysis.

Notes:
This basic version uses histogram features, but real-world systems use Gabor wavelets, phase encoding, and segmentation for sclera and pupil.

Accuracy improves with iris segmentation and noise filtering (e.g., eyelids, reflections).
"""

import cv2
import numpy as np
from scipy.spatial.distance import cosine
 
# Function to preprocess iris image (grayscale + cropping + resize)
def preprocess_iris(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    # Normalize intensity and flatten to feature vector
    norm_img = cv2.equalizeHist(img)
    return norm_img.flatten()
 
# Extract iris features (basic pixel histogram + shape descriptor)
def extract_features(flattened_img):
    hist = np.histogram(flattened_img, bins=32, range=(0, 256))[0]
    hist = hist / np.linalg.norm(hist)  # Normalize histogram
    return hist
 
# Preprocess and extract features
enrolled_img = preprocess_iris("enrolled_iris.jpg")
input_img = preprocess_iris("input_iris.jpg")
 
feature1 = extract_features(enrolled_img)
feature2 = extract_features(input_img)
 
# Compare features using cosine similarity
similarity = 1 - cosine(feature1, feature2)
threshold = 0.85  # Typical threshold for match
 
print(f"Iris Similarity Score: {similarity:.2f}")
if similarity >= threshold:
    print("✅ Iris Verified: Access Granted.")
else:
    print("❌ Iris Mismatch: Access Denied.")