"""
Project 928. Cross-modal Retrieval System

A cross-modal retrieval system allows users to retrieve information from one modality (e.g., text) based on queries from another modality (e.g., image). In this project, we simulate a simple image-to-text retrieval system, where we retrieve relevant text descriptions based on input images using pre-trained multi-modal models.

What This Does:
CLIP (Contrastive Language-Image Pre-training) is used to align image and text embeddings in a shared space.

The system retrieves the most relevant text description for a given image by comparing their embeddings.
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
 
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulate a text and image dataset
texts = ["A picture of a cat", "A picture of a dog", "A beautiful landscape"]
images = ["cat_image.jpg", "dog_image.jpg", "landscape_image.jpg"]  # replace with valid image paths
 
# Process image and text queries
def cross_modal_retrieval(query_image, query_texts):
    image = Image.open(query_image)
    inputs = processor(text=query_texts, images=image, return_tensors="pt", padding=True)
 
    # Get similarity between the image and texts
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Image-text similarity scores
    probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities
 
    # Retrieve the most relevant text for the image
    best_match_idx = torch.argmax(probs)
    return query_texts[best_match_idx], probs[0][best_match_idx].item()
 
# Simulate a query image (replace with the actual query image path)
query_image_path = "cat_image.jpg"  # Example query image
matched_text, score = cross_modal_retrieval(query_image_path, texts)
 
# Output the most relevant text description for the image
print(f"Query Image: {query_image_path}")
print(f"Most Relevant Text: {matched_text}")
print(f"Match Score: {score:.2f}")