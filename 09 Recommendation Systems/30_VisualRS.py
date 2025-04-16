"""
Project 350. Visual recommendation system
Description:
A visual recommendation system uses image data (such as product photos or visual content) to enhance recommendations. This can be particularly useful for:

E-commerce platforms that recommend products based on their visual similarity (e.g., clothing, furniture)

Content-based recommendations that rely on visual features like color, style, or shape

In this project, we’ll create a visual recommendation system by extracting visual features from product images using pre-trained Convolutional Neural Networks (CNNs).

About:
✅ What It Does:
Pre-trained ResNet model extracts visual features from product images

Cosine similarity is used to find items that are visually similar to a query item

Provides visual recommendations based on image content (e.g., recommending products similar to a laptop based on its image)
"""
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
 
# 1. Simulate items with image files (we'll use random image paths as an example)
items = ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch']
image_paths = {
    'Laptop': 'path_to_laptop_image.jpg',
    'Smartphone': 'path_to_smartphone_image.jpg',
    'Headphones': 'path_to_headphones_image.jpg',
    'Tablet': 'path_to_tablet_image.jpg',
    'Smartwatch': 'path_to_smartwatch_image.jpg'
}
 
# 2. Define the image transformation (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to match input size for pre-trained model
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained model
])
 
# 3. Load a pre-trained ResNet model (for feature extraction)
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode (we don't need gradients for inference)
 
# 4. Function to extract visual features from images
def extract_visual_features(image_path):
    image = Image.open(image_path).convert('RGB')  # Open the image and ensure it's in RGB format
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
    
    with torch.no_grad():  # No need to compute gradients for feature extraction
        features = model(image)
    
    return features.flatten()  # Flatten the feature vector for easier comparison
 
# 5. Function to recommend items based on visual similarity (cosine similarity)
def visual_recommendation(query_item, image_paths, top_n=3):
    # Extract visual features for the query item
    query_features = extract_visual_features(image_paths[query_item])
    
    similarities = []
    for item, path in image_paths.items():
        if item != query_item:  # Skip the query item itself
            item_features = extract_visual_features(path)
            similarity = torch.nn.functional.cosine_similarity(query_features, item_features)
            similarities.append((item, similarity.item()))
    
    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top N recommended items based on visual similarity
    return [item for item, _ in similarities[:top_n]]
 
# 6. Recommend items based on the visual similarity of 'Laptop'
recommended_items = visual_recommendation('Laptop', image_paths)
print(f"Visual Recommendations for Laptop: {recommended_items}")