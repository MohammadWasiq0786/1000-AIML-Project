"""
Project 895. Behavioral Biometrics System

Behavioral biometrics analyzes patterns in how users interact with devices — such as typing rhythm, mouse movement, or swipe gestures — to continuously verify identity. In this project, we simulate keystroke dynamics (timing between key presses) and use Euclidean distance for identity matching.

What It Does:
Captures flight time (time between key presses) or dwell time (time a key is held).

Uses distance metric to compare current behavior with enrolled template.

🛡️ Advanced behavioral biometrics may include:

Mouse trajectory modeling

Mobile swipe/gesture patterns

Continuous authentication over a session
"""

import numpy as np
from scipy.spatial.distance import euclidean
 
# Simulated keystroke timing profile (in milliseconds between keys)
# Reference typing pattern (enrolled user)
enrolled_profile = np.array([120, 100, 130, 150, 110])  # e.g., password "hello"
 
# Input pattern (captured during login)
input_profile = np.array([118, 102, 129, 152, 111])  # same password typed again
 
# Compare profiles using Euclidean distance
distance = euclidean(enrolled_profile, input_profile)
threshold = 15  # Acceptable variation in timing
 
print(f"Keystroke Distance: {distance:.2f}")
if distance < threshold:
    print("✅ Behavioral Biometrics Match: Identity Verified.")
else:
    print("❌ Behavior Mismatch: Access Denied.")