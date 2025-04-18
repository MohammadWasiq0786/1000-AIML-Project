"""
Project 929. Multi-modal Recommendation System

A multi-modal recommendation system leverages multiple types of data (e.g., text, image, and user interaction data) to provide personalized recommendations. In this project, we simulate a recommendation system that uses user preferences (text reviews) and product images to suggest items to a user.

What This Does:
CLIP is used to create embeddings for both the product images and user preferences (text and image).

The system calculates the cosine similarity between the user's preferences and the products, ranking products based on similarity.
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
 
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulated product dataset: product images and reviews
products = [
    {"image": "product1.jpg", "text": "A stylish pair of running shoes."},
    {"image": "product2.jpg", "text": "A comfy cotton t-shirt."},
    {"image": "product3.jpg", "text": "A cozy winter jacket."},
    {"image": "product4.jpg", "text": "A sleek smartwatch with fitness tracker."}
]
 
# User's preference (text review and image of the preferred item)
user_review = "I love running shoes, especially those with great support and comfort."
user_image = "user_preferred_image.jpg"  # Replace with user preference image
 
# Function to retrieve top N recommended products based on user's preferences
def recommend_products(user_review, user_image, products, top_n=3):
    # Process user's text and image
    inputs = processor(text=[user_review] * len(products), images=[Image.open(p['image']) for p in products], return_tensors="pt", padding=True)
    
    # Process user's query (text and image)
    user_inputs = processor(text=[user_review] * len(products), images=[Image.open(user_image)] * len(products), return_tensors="pt", padding=True)
 
    # Get similarity scores between user preferences and products
    outputs = model(**inputs)
    user_outputs = model(**user_inputs)
 
    # Compute similarity (dot product of text-image embeddings)
    text_image_similarity = torch.cosine_similarity(outputs.text_embeds, user_outputs.text_embeds)
 
    # Rank products based on similarity scores
    scores = text_image_similarity.cpu().detach().numpy()
    recommended_idx = np.argsort(scores)[-top_n:][::-1]  # Get top N recommendations
 
    # Return top N recommended products
    return [products[i] for i in recommended_idx]
 
# Get top N recommended products
recommended_products = recommend_products(user_review, user_image, products, top_n=2)
 
# Display recommended products
print("Recommended Products:")
for product in recommended_products:
    print(f"Product: {product['text']}")