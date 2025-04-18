"""
Project 941. Multi-modal Self-supervised Learning

Self-supervised learning (SSL) is a type of unsupervised learning where the model learns useful representations from the data without relying on labeled data. Multi-modal self-supervised learning involves learning representations that combine information from multiple modalities, such as text, images, and audio, without explicit supervision.

In this project, we simulate a multi-modal self-supervised learning task by training a model to learn representations from both text and image data. We'll use the CLIP model for self-supervised learning, where the model learns to correlate text and image features.

Step 1: Data Preprocessing
We’ll use a contrastive loss function (similar to what CLIP uses) to learn representations of both images and text.

Step 2: Self-supervised Learning
Instead of using labeled data, we train the model by comparing whether a text and an image belong together (positive pair) or not (negative pair).

What This Does:
Self-supervised Learning: The model learns to associate images and texts based on their similarity without any labeled data.

Contrastive Loss: We use a contrastive loss function to train the model to bring matching image-text pairs closer together and non-matching pairs farther apart in the embedding space.

Training Setup: This demo simulates how a model can be trained using self-supervised learning with image-text pairs.
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
 
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulate a dataset of image-text pairs (self-supervised task)
image_text_pairs = [
    {"image": "dog_image.jpg", "text": "A picture of a dog."},
    {"image": "cat_image.jpg", "text": "A picture of a cat."},
    {"image": "car_image.jpg", "text": "A picture of a car."},
    {"image": "flower_image.jpg", "text": "A picture of a flower."}
]
 
# Simulate a batch of text and image data for self-supervised learning
images = [Image.open(item['image']) for item in image_text_pairs]
texts = [item['text'] for item in image_text_pairs]
 
# Preprocess the images and text
inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
 
# Perform a forward pass to get model's image-text embeddings
outputs = model(**inputs)
image_embeddings = outputs.image_embeds
text_embeddings = outputs.text_embeds
 
# Step 1: Contrastive loss function for self-supervised learning
def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    similarity_matrix = torch.matmul(image_embeds, text_embeds.T)
    labels = torch.arange(image_embeds.size(0)).to(image_embeds.device)
    loss = torch.nn.CrossEntropyLoss()(similarity_matrix / temperature, labels)
    return loss
 
# Step 2: Compute loss and simulate model optimization
loss = contrastive_loss(image_embeddings, text_embeddings)
print(f"Contrastive Loss: {loss.item():.4f}")
 
# Simulate self-supervised learning (in practice, this would involve backpropagation and optimization)
# For this demo, we're just showing how loss is calculated for a batch of image-text pairs