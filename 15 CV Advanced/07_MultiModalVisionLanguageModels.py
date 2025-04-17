"""
Project 567: Multi-modal Vision-Language Models
Description:
Multi-modal vision-language models combine visual information (e.g., images, videos) with textual data to enable more advanced understanding and interaction with the world. These models are capable of tasks like image captioning, visual question answering, and image-text alignment. In this project, we will use pre-trained models such as CLIP or VisualBERT to perform tasks that require understanding both images and text.
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
 
# 1. Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# 2. Load an image to be processed
image = Image.open("path_to_image.jpg")  # Replace with an actual image path
 
# 3. Define candidate text labels
text = ["a photo of a cat", "a photo of a dog", "a picture of a person"]
 
# 4. Preprocess the image and text
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
 
# 5. Perform zero-shot classification
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # Image-text similarity
probs = logits_per_image.softmax(dim=1)  # Get the probabilities
 
# 6. Display the result
label = text[torch.argmax(probs)]
print(f"Predicted label: {label} with confidence {100 * torch.max(probs).item():.2f}%")