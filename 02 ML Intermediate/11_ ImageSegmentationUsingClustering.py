"""
Project 51. Image segmentation using clustering
Description:
Image segmentation is the process of partitioning an image into meaningful segments (like separating objects from background). In this project, we apply K-Means clustering to segment an image based on pixel color similarity — effectively grouping similar colors and simplifying the image.

About:
🧠 What This Project Demonstrates:
Reshapes image data for pixel-wise clustering

Applies K-Means to segment an image into color clusters

Visualizes a simplified version of the image based on dominant colors
"""

# Install OpenCV if needed: pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
# Load the image and convert it to RGB
image = cv2.imread('sample_image.jpg')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_shape = image.shape
 
# Reshape image into a 2D array of pixels (rows: pixels, cols: RGB channels)
pixel_data = image.reshape((-1, 3))
 
# Apply K-Means clustering
k = 4  # Number of clusters/segments
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixel_data)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
 
# Reshape back to original image dimensions
segmented_img = segmented_img.reshape(original_shape).astype(np.uint8)
 
# Display original vs segmented image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title(f"Segmented Image (K={k})")
plt.axis('off')
 
plt.tight_layout()
plt.show()