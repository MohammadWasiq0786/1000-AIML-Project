"""
Project 890. Biometric Verification System

A biometric verification system authenticates users based on unique biological traits like fingerprint, face, or voice. In this project, we simulate a fingerprint matching system using vectorized biometric features and compare them using Euclidean distance.

Key Points:
This simulates matching of biometric vectors, commonly extracted from images using CNNs in real systems.

The threshold determines how strict the verification is.

You can replace vectors with actual biometric model outputs using OpenCV, TensorFlow, or pretrained APIs (e.g., face_recognition or DeepFace).
"""

import numpy as np
from scipy.spatial.distance import euclidean
 
# Simulated biometric fingerprint vectors (128-dim embeddings)
# Stored template for enrolled user
registered_fingerprint = np.random.rand(128)
 
# New fingerprint captured during login
new_fingerprint = registered_fingerprint + np.random.normal(0, 0.01, 128)  # slight noise
 
# Compare using Euclidean distance
distance = euclidean(registered_fingerprint, new_fingerprint)
 
# Define threshold (tunable based on system sensitivity)
threshold = 0.5
is_verified = distance < threshold
 
# Output results
print(f"Fingerprint Distance: {distance:.4f}")
if is_verified:
    print("✅ Biometric Match: Access granted.")
else:
    print("❌ Biometric Mismatch: Access denied.")