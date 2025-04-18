"""
Project 945. Cross-modal Alignment Techniques

Cross-modal alignment techniques aim to map data from different modalities (such as text, images, audio) into a shared embedding space. The goal is to ensure that related data points (e.g., a text description and the corresponding image) are close together in the shared space, facilitating tasks like cross-modal retrieval, image captioning, and text-based image generation.

In this project, we will focus on aligning text and images using a contrastive loss function to ensure that related image-text pairs are aligned in a common feature space.

Step 1: Text and Image Embedding
We will use CLIP to get embeddings for both text and images.

Step 2: Cross-modal Alignment
We will compute the cosine similarity between text and image embeddings and use a contrastive loss function to align them in the shared space.

What This Does:
Cross-modal Embedding: Uses CLIP to get embeddings for both images and texts.

Contrastive Loss: A contrastive loss function is used to align image-text pairs by minimizing the distance between embeddings of related image-text pairs and maximizing the distance for unrelated ones.

Cross-modal Retrieval: Retrieves the most relevant images based on a text query by calculating the cosine similarity between the text query's embedding and the image embeddings
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
 
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulated image-text dataset for cross-modal alignment
image_text_pairs = [
    {"image": "dog_image.jpg", "text": "A picture of a dog."},
    {"image": "cat_image.jpg", "text": "A picture of a cat."},
    {"image": "car_image.jpg", "text": "A picture of a car."},
    {"image": "flower_image.jpg", "text": "A picture of a flower."}
]
 
# Step 1: Preprocess the image and text data to get embeddings
images = [Image.open(item['image']) for item in image_text_pairs]
texts = [item['text'] for item in image_text_pairs]
 
inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
 
# Perform forward pass through the CLIP model to get image-text embeddings
outputs = model(**inputs)
image_embeddings = outputs.image_embeds
text_embeddings = outputs.text_embeds
 
# Step 2: Cross-modal Alignment - Contrastive loss function to align image and text representations
def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    # Calculate similarity between image-text pairs
    similarity_matrix = torch.matmul(image_embeds, text_embeds.T)
    labels = torch.arange(image_embeds.size(0)).to(image_embeds.device)
    
    # Use cross-entropy loss to encourage correct alignment
    loss = torch.nn.CrossEntropyLoss()(similarity_matrix / temperature, labels)
    return loss
 
# Compute the contrastive loss for cross-modal alignment
loss = contrastive_loss(image_embeddings, text_embeddings)
print(f"Contrastive Loss for Cross-modal Alignment: {loss.item():.4f}")
 
# After alignment, we can perform cross-modal tasks like image-text retrieval:
def retrieve_images_from_text(query, image_embeddings, text_embeddings, top_n=2):
    query_inputs = processor(text=[query] * len(image_embeddings), images=images, return_tensors="pt", padding=True)
    query_outputs = model(**query_inputs)
    query_text_embeddings = query_outputs.text_embeds
 
    # Compute similarity between the query and image embeddings
    similarity_scores = torch.cosine_similarity(query_text_embeddings, image_embeddings)
    best_match_idx = torch.argsort(similarity_scores, descending=True)[:top_n]
 
    return [texts[i] for i in best_match_idx], [images[i] for i in best_match_idx]
 
# Example: Retrieve images based on a text query
query = "A picture of a dog"
retrieved_texts, retrieved_images = retrieve_images_from_text(query, image_embeddings, text_embeddings)
 
print(f"Text Query: {query}")
print(f"Most Relevant Texts: {retrieved_texts}")
print(f"Most Relevant Images: {retrieved_images}")