"""
Project 572: Referring Expression Comprehension
Description:
Referring expression comprehension involves identifying and understanding objects in an image based on natural language descriptions. For example, given the phrase "the red ball on the table," the model needs to locate the red ball in the image. In this project, we will use a pre-trained vision-and-language model to perform referring expression comprehension.
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
 
# 1. Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# 2. Load an image for referring expression comprehension
image = Image.open("path_to_image.jpg")  # Replace with an actual image path
 
# 3. Define a referring expression (e.g., "the red ball on the table")
referring_expression = "the red ball on the table"
 
# 4. Preprocess the image and referring expression
inputs = processor(text=[referring_expression], images=image, return_tensors="pt", padding=True)
 
# 5. Perform referring expression comprehension (image-text similarity)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # Image-text similarity
probs = logits_per_image.softmax(dim=1)  # Get the probabilities
 
# 6. Display the result
print(f"Referring Expression Comprehension: {referring_expression} with confidence {100 * torch.max(probs).item():.2f}%")