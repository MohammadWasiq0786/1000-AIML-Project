"""
Project 67. Image similarity search engine
Description:
An image similarity search engine lets you find images visually similar to a given input. It works by extracting feature vectors from images and comparing them using a similarity metric. In this project, we use pre-trained CNN features from ResNet50 and cosine similarity to match similar images.

About:
🖼️ What This Project Demonstrates:
Extracts deep features from images using ResNet50

Calculates cosine similarity to find visually similar images

Builds the core of an image search or recommendation engine
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
 
# Load pre-trained ResNet50 model (excluding the final classification layer)
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
 
# Helper function: preprocess and extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Load image
    img_array = image.img_to_array(img)                     # Convert to array
    img_array = np.expand_dims(img_array, axis=0)           # Add batch dimension
    img_array = preprocess_input(img_array)                 # Preprocess for ResNet
    features = model.predict(img_array, verbose=0)          # Extract features
    return features[0]                                      # Return feature vector
 
# Load image dataset (sample folder of images)
image_folder = 'images/'  # Replace with your folder path
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]
 
# Extract feature vectors for all images
feature_list = []
for path in image_paths:
    feature = extract_features(path)
    feature_list.append(feature)
 
# Choose a query image to compare against the dataset
query_image_path = image_paths[0]
query_feature = extract_features(query_image_path)
 
# Calculate cosine similarity between query and dataset
similarities = cosine_similarity([query_feature], feature_list)[0]
top_indices = similarities.argsort()[::-1][1:6]  # Top 5 similar (excluding itself)
 
# Display query image and top matches
plt.figure(figsize=(12, 3))
plt.subplot(1, 6, 1)
plt.imshow(cv2.cvtColor(cv2.imread(query_image_path), cv2.COLOR_BGR2RGB))
plt.title("Query")
plt.axis('off')
 
for i, idx in enumerate(top_indices):
    img_path = image_paths[idx]
    img = cv2.imread(img_path)
    plt.subplot(1, 6, i+2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Sim {similarities[idx]:.2f}")
    plt.axis('off')
 
plt.suptitle("Image Similarity Search Results", fontsize=14)
plt.tight_layout()
plt.show()