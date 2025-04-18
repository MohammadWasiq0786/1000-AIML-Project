"""
Project 944. Multi-modal Representation Learning

Multi-modal representation learning involves learning a unified representation space for multiple modalities (e.g., text, image, audio) that can be jointly used for downstream tasks such as classification, retrieval, or generation. In this project, we simulate the learning of shared representations from both images and texts using the CLIP model. The goal is to learn a combined representation that captures both image content and textual information in a common feature space.

Step 1: Learning Image and Text Representations
We’ll use CLIP to learn embeddings from both images and text and project them into a shared embedding space.

Step 2: Multi-modal Retrieval
After learning the representations, we’ll retrieve relevant images based on text queries (and vice versa) by calculating similarities between the embeddings.

What This Does:
Image and Text Representation: The model CLIP processes both text and images and learns embeddings in a shared space.

Multi-modal Retrieval: It calculates the cosine similarity between the text query and image embeddings to retrieve the most relevant images for the query (and vice versa).

Multi-modal Alignment: The system learns a joint representation that enables comparison across modalities (text and images).
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
 
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulated image-text dataset
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
 
# Step 2: Multi-modal Retrieval: Retrieve the most relevant image for a given text query
def retrieve_images_by_text_query(query, image_embeddings, text_embeddings, top_n=2):
    query_inputs = processor(text=[query] * len(image_embeddings), images=images, return_tensors="pt", padding=True)
    query_outputs = model(**query_inputs)
    query_text_embeddings = query_outputs.text_embeds
 
    # Compute similarity between the query and image embeddings
    similarity_scores = torch.cosine_similarity(query_text_embeddings, image_embeddings)
    best_match_idx = torch.argsort(similarity_scores, descending=True)[:top_n]
 
    return [texts[i] for i in best_match_idx], [images[i] for i in best_match_idx]
 
# Example: Retrieve images based on a text query
query = "A picture of a dog"
retrieved_texts, retrieved_images = retrieve_images_by_text_query(query, image_embeddings, text_embeddings)
 
print(f"Text Query: {query}")
print(f"Most Relevant Texts: {retrieved_texts}")
print(f"Most Relevant Images: {retrieved_images}")