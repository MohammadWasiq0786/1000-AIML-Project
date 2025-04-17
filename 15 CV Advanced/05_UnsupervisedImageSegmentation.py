"""
Project 565: Unsupervised Image Segmentation
Description:
Unsupervised image segmentation involves dividing an image into meaningful segments or regions without using labeled data. This task is often used to identify objects or regions of interest within images. In this project, we will use clustering techniques such as K-means or Self-organizing maps (SOM) to perform unsupervised segmentation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
# 1. Load the image
image = cv2.imread("path_to_image.jpg")  # Replace with an actual image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# 2. Reshape the image for clustering
pixels = image.reshape(-1, 3)  # Reshape to (num_pixels, num_channels)
 
# 3. Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(pixels)
 
# 4. Reshape the cluster labels to match the original image shape
segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
 
# 5. Display the original and segmented image
plt.figure(figsize=(10, 5))
 
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
 
plt.subplot(1, 2, 2)
plt.imshow(segmented_image.astype(np.uint8))
plt.title("Segmented Image")
 
plt.show()