"""
Project 943. Multi-modal Domain Adaptation

Domain adaptation in multi-modal systems involves transferring a model trained on one domain to a different but related domain, adapting it to new data with minimal supervision. This allows the model to generalize across domains, such as applying a model trained on general images and text to a specialized domain like medical images or e-commerce.

In this project, we simulate multi-modal domain adaptation by adapting a pre-trained model to a new domain using image-text pairs. We'll focus on adapting a model trained on general datasets to work with a new domain (e.g., medical images or e-commerce product descriptions).

Step 1: Pre-trained Model
We'll use CLIP to learn general representations from a general domain (e.g., generic images and descriptions).

Step 2: Domain Adaptation
We simulate domain adaptation by fine-tuning the pre-trained model on a new domain (e.g., medical images and medical descriptions).

What This Does:
Pre-trained CLIP Model: We simulate transfer from a general dataset (e.g., everyday images and text) to a new domain (e.g., medical images and text).

Domain Adaptation: We fine-tune the model on a new domain (medical data) using a contrastive loss function to align image-text representations from the new domain.

Prediction: After fine-tuning, the model can classify new images based on the adapted domain knowledge.
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
 
# Load pre-trained CLIP model and processor (trained on a general dataset)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulate a new domain: medical images and descriptions
medical_data = [
    {"image": "medical_image1.jpg", "text": "An X-ray showing signs of pneumonia."},
    {"image": "medical_image2.jpg", "text": "A CT scan of a healthy lung."},
    {"image": "medical_image3.jpg", "text": "MRI showing brain tumor."},
    {"image": "medical_image4.jpg", "text": "An ultrasound image of a healthy liver."}
]
 
# Simulate pre-trained general dataset (image-text pairs)
general_data = [
    {"image": "dog_image.jpg", "text": "A picture of a dog."},
    {"image": "cat_image.jpg", "text": "A picture of a cat."},
    {"image": "car_image.jpg", "text": "A picture of a car."},
    {"image": "flower_image.jpg", "text": "A picture of a flower."}
]
 
# Pre-process and extract embeddings for the general domain (pre-trained model)
general_images = [Image.open(item['image']) for item in general_data]
general_texts = [item['text'] for item in general_data]
general_inputs = processor(text=general_texts, images=general_images, return_tensors="pt", padding=True)
 
# Perform forward pass through the pre-trained model
general_outputs = model(**general_inputs)
general_image_embeddings = general_outputs.image_embeds
general_text_embeddings = general_outputs.text_embeds
 
# Simulate domain adaptation by training on new medical data (new domain)
medical_images = [Image.open(item['image']) for item in medical_data]
medical_texts = [item['text'] for item in medical_data]
medical_inputs = processor(text=medical_texts, images=medical_images, return_tensors="pt", padding=True)
 
# Perform forward pass on the new domain data
medical_outputs = model(**medical_inputs)
medical_image_embeddings = medical_outputs.image_embeds
medical_text_embeddings = medical_outputs.text_embeds
 
# Domain adaptation: Use contrastive loss to fine-tune the model for the new medical domain
def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    similarity_matrix = torch.matmul(image_embeds, text_embeds.T)
    labels = torch.arange(image_embeds.size(0)).to(image_embeds.device)
    loss = torch.nn.CrossEntropyLoss()(similarity_matrix / temperature, labels)
    return loss
 
# Compute contrastive loss for the new medical domain
loss = contrastive_loss(medical_image_embeddings, medical_text_embeddings)
print(f"Domain Adaptation Loss (Medical Domain): {loss.item():.4f}")
 
# After fine-tuning, we simulate classification or retrieval based on adapted model
# Example: Use the adapted model to classify a new medical image (e.g., a medical diagnosis)
new_image = Image.open("new_medical_image.jpg")  # Replace with a valid image path
new_texts = ["An X-ray showing signs of pneumonia.", "A CT scan of a healthy lung."]
new_inputs = processor(text=new_texts, images=[new_image] * len(new_texts), return_tensors="pt", padding=True)
 
# Perform forward pass to classify new image
new_outputs = model(**new_inputs)
new_image_embeddings = new_outputs.image_embeds
new_text_embeddings = new_outputs.text_embeds
 
# Calculate similarity between new image and the text descriptions
similarity_scores = torch.cosine_similarity(new_image_embeddings, new_text_embeddings)
best_match_idx = torch.argmax(similarity_scores)
 
# Output the predicted class for the new medical image
print(f"Predicted Class for New Medical Image: {new_texts[best_match_idx]}")