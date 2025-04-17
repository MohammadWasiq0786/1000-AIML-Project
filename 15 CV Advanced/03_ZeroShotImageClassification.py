"""
Project 563: Zero-shot Image Classification
Description:
Zero-shot image classification allows a model to classify images into categories without having seen any labeled data from those categories during training. Instead, it uses a semantic understanding of the categories (e.g., word embeddings or descriptions). In this project, we will implement zero-shot image classification using pre-trained models like CLIP (Contrastive Language-Image Pretraining).
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
 
# 1. Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# 2. Load the image to classify
image = Image.open("path_to_image.jpg")  # Replace with an actual image path
 
# 3. Define candidate labels for classification
labels = ["a photo of a dog", "a photo of a cat", "a photo of a person"]
 
# 4. Preprocess the image and the labels
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
 
# 5. Perform zero-shot classification
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity
probs = logits_per_image.softmax(dim=1) # we can get the label probabilities
 
# 6. Display the result
label = labels[torch.argmax(probs)]
print(f"Predicted label: {label} with confidence {100 * torch.max(probs).item():.2f}%")