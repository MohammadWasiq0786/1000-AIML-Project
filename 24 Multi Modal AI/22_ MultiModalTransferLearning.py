"""
Project 942. Multi-modal Transfer Learning

Multi-modal transfer learning involves leveraging pre-trained models from one modality (e.g., text or image) to enhance learning in another modality, often with less data. In this project, we simulate transfer learning across text and images using a pre-trained CLIP model. The model learns to transfer knowledge from one modality to another, allowing it to perform tasks such as image classification based on text or text generation based on image context.

Step 1: Pre-trained Model
We use the CLIP model for transfer learning across images and text. We'll first use it to get embeddings for both image and text data, and then apply the learned embeddings to a new task (e.g., image classification or text-based image retrieval).

Step 2: Fine-tuning for Transfer Learning
We simulate fine-tuning the model for a new task, such as classifying images into categories based on text prompts.

What This Does:
Pre-trained Model: We use the CLIP model pre-trained on both text and images for transfer learning.

Contrastive Loss: We simulate fine-tuning by calculating a contrastive loss between image and text embeddings.

Transfer Learning: The model transfers knowledge from image-text matching to classify new images based on a few-shot learning approach.
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
 
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulate a dataset of image-text pairs for transfer learning
image_text_pairs = [
    {"image": "dog_image.jpg", "text": "A picture of a dog."},
    {"image": "cat_image.jpg", "text": "A picture of a cat."},
    {"image": "car_image.jpg", "text": "A picture of a car."},
    {"image": "flower_image.jpg", "text": "A picture of a flower."}
]
 
# Preprocess the image and text inputs
images = [Image.open(item['image']) for item in image_text_pairs]
texts = [item['text'] for item in image_text_pairs]
 
inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
 
# Perform a forward pass to get model's image-text embeddings
outputs = model(**inputs)
image_embeddings = outputs.image_embeds
text_embeddings = outputs.text_embeds
 
# Step 1: Simulate fine-tuning by calculating similarity (using image-text contrastive loss)
def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    similarity_matrix = torch.matmul(image_embeds, text_embeds.T)
    labels = torch.arange(image_embeds.size(0)).to(image_embeds.device)
    loss = torch.nn.CrossEntropyLoss()(similarity_matrix / temperature, labels)
    return loss
 
# Step 2: Compute the loss and simulate transfer learning
loss = contrastive_loss(image_embeddings, text_embeddings)
print(f"Contrastive Loss for Transfer Learning: {loss.item():.4f}")
 
# Step 3: Transfer learning for new image classification task
# Simulate that we are transferring the knowledge from CLIP's learned embeddings to a new task:
# Example: Given a new image, classify it using the text-to-image embeddings
 
new_image = Image.open("new_image.jpg")  # Replace with a valid image path
new_texts = ["A picture of a dog.", "A picture of a cat.", "A picture of a car."]
 
# Process new data
new_inputs = processor(text=new_texts, images=[new_image] * len(new_texts), return_tensors="pt", padding=True)
 
# Forward pass through the model for classification
new_outputs = model(**new_inputs)
new_image_embeddings = new_outputs.image_embeds
new_text_embeddings = new_outputs.text_embeds
 
# Calculate similarity between new image and the text descriptions
similarity_scores = torch.cosine_similarity(new_image_embeddings, new_text_embeddings)
best_match_idx = torch.argmax(similarity_scores)
 
# Output the best match (classified label)
print(f"Predicted Class for New Image: {new_texts[best_match_idx]}")