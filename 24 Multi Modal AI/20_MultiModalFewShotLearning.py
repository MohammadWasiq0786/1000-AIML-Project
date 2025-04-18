"""
Project 940. Multi-modal Few-shot Learning

Few-shot learning allows a model to generalize from a very small number of training examples. Multi-modal few-shot learning extends this concept by combining different modalities (e.g., text, image, audio) to enable a model to make predictions or decisions with minimal data from each modality.

In this project, we simulate a few-shot learning task by training a multi-modal model (using both images and text) to classify items based on just a few examples.

Step 1: Few-shot Learning on Image and Text
We use the CLIP model to create embeddings for both images and text. We then simulate a few-shot classification task where the model uses just a few examples to classify new items.

Step 2: Matching New Data
For this example, we simulate classifying an image of a dog using a few labeled examples (like "This is a dog" for text and an image of a dog).

What This Does:
Few-shot Learning: The model learns from just a few labeled examples (images and text descriptions) to classify new data (an image of a dog, in this case).

Multi-modal Matching: Uses CLIP to create embeddings for both the images and texts and calculates the cosine similarity between the new image and the few-shot examples.

Prediction: The model predicts the class of the new image based on the closest match in the few-shot training data.
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
 
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulated few-shot labeled data (image-text pairs)
few_shot_data = [
    {"image": "dog_image.jpg", "text": "This is a dog."},
    {"image": "cat_image.jpg", "text": "This is a cat."}
]
 
# Simulate new image to classify (replace with a valid image path)
new_image = Image.open("new_image.jpg")  # New image to classify (e.g., an image of a dog)
 
# Preprocess the few-shot text and images
few_shot_texts = [item['text'] for item in few_shot_data]
few_shot_images = [Image.open(item['image']) for item in few_shot_data]
inputs = processor(text=few_shot_texts, images=few_shot_images, return_tensors="pt", padding=True)
 
# Preprocess the new image for classification
new_image_input = processor(text=["What is this?"], images=[new_image], return_tensors="pt", padding=True)
 
# Perform forward pass for few-shot learning and classification
outputs = model(**inputs)
new_image_outputs = model(**new_image_input)
 
# Calculate similarity between new image and few-shot images
logits_per_image = outputs.logits_per_image  # Image-text similarity scores for few-shot images
logits_new_image = new_image_outputs.logits_per_image  # Similarity score for new image
 
# Calculate cosine similarity between new image and few-shot images
similarity_scores = torch.cosine_similarity(logits_per_image, logits_new_image)
best_match_idx = torch.argmax(similarity_scores)
 
# Output the predicted class (text description) for the new image
predicted_class = few_shot_texts[best_match_idx]
print(f"Predicted Class for New Image: {predicted_class} with similarity score {similarity_scores[best_match_idx]:.2f}")