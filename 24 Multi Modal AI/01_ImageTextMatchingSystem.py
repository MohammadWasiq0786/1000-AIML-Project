"""
Project 921. Image-Text Matching System

An image-text matching system is designed to associate images with corresponding textual descriptions. It helps in applications like image search, visual question answering, and multi-modal retrieval. In this project, we simulate a basic system where the goal is to match images with text using feature embeddings.

What This Does:
CLIP (Contrastive Language-Image Pre-training) is used to match images with textual descriptions.

It computes similarity between the image and the text, helping us identify the most relevant description for an image.
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
 
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulate image and text data
image = Image.open("example_image.jpg")  # Replace with a valid image path
texts = ["A photo of a cat", "A photo of a dog", "A beautiful landscape"]
 
# Preprocess the image and text inputs
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
 
# Perform forward pass
outputs = model(**inputs)
 
# Calculate similarity between the image and each text
logits_per_image = outputs.logits_per_image # Image-text similarity scores
probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities
 
# Display the matching results
print("Image-Text Matching Results:")
for i, text in enumerate(texts):
    print(f"Text: {text} | Probability: {probs[0][i]:.4f}")
