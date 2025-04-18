"""
Project 969: Zero-shot Learning Implementation
Description
Zero-shot learning allows a model to correctly make predictions on tasks it has never seen before, based solely on descriptions or semantic information about the classes. In this project, we will implement a simple zero-shot learning system for image classification, where the model predicts labels for unseen classes based on textual descriptions.

Key Concepts Covered:
Zero-shot Learning (ZSL): The ability of a model to classify instances from classes it has never seen before, using textual or semantic information.

CLIP Model: A pre-trained model that learns to associate images with text descriptions. It can be used for zero-shot classification by comparing image and text embeddings.

Cosine Similarity: Measures the cosine of the angle between two vectors in the feature space to determine their similarity.
"""

# pip install torch torchvision clip-by-openai

import torch
import clip
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
 
# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)
 
# Example classes and their descriptions (Zero-shot learning)
class_descriptions = [
    "A photo of a cat",
    "A photo of a dog",
    "A picture of a car",
    "A picture of a tree"
]
 
# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
# Load an image to classify
image_path = 'example_image.jpg'  # Replace with your image path
image = Image.open(image_path)
image_input = preprocess(image).unsqueeze(0).to(device)
 
# Encode the text descriptions using CLIP
text_inputs = torch.cat([clip.tokenize(desc).unsqueeze(0) for desc in class_descriptions]).to(device)
 
# Generate image and text features using CLIP
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
 
# Normalize the features
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
 
# Calculate similarity scores (cosine similarity)
similarity = (image_features @ text_features.T).squeeze(0)
 
# Get the index of the highest similarity (the predicted class)
predicted_class_idx = np.argmax(similarity.cpu().numpy())
print(f"Predicted class: {class_descriptions[predicted_class_idx]}")