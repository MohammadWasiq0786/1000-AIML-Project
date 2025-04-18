"""
Project 892. Fingerprint Recognition

Fingerprint recognition systems match a scanned fingerprint against a database to verify or identify a person. In this project, we simulate fingerprint recognition using image matching techniques with OpenCV’s ORB (Oriented FAST and Rotated BRIEF) descriptor and feature matching.

Key Concepts:
ORB is fast, lightweight, and works well for biometric feature extraction.

The lower the average match distance, the closer the fingerprints.

Threshold tuning is critical based on dataset quality and resolution.
"""

import cv2
 
# Load registered and scanned fingerprint images (grayscale)
img1 = cv2.imread("registered_fingerprint.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("scanned_fingerprint.jpg", cv2.IMREAD_GRAYSCALE)
 
# Initialize ORB detector
orb = cv2.ORB_create()
 
# Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
 
# Match features using Brute-Force Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
 
# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)
 
# Calculate average distance of top N matches
N = 30
average_distance = sum([m.distance for m in matches[:N]]) / N
threshold = 50  # Lower = more similar
 
# Display result
print(f"Average Match Distance: {average_distance:.2f}")
if average_distance < threshold:
    print("✅ Fingerprint Match: Access granted.")
else:
    print("❌ Fingerprint Mismatch: Access denied.")
 
# Optionally, visualize matches
matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:N], None, flags=2)
cv2.imshow("Matched Fingerprints", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()