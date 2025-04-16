"""
Project 211. Image captioning model
Description:
Image captioning is the task of generating a natural language description for an image. It combines computer vision and natural language processing. Typically, a CNN (like ResNet) extracts visual features, and an RNN (like LSTM) or a Transformer decodes those features into a caption. We'll implement a simple version using a pre-trained image captioning pipeline from Hugging Face Transformers.

About:
What It Does:
This project takes any image and produces a descriptive sentence like "A dog is running in the park." It's used in accessibility tools for the visually impaired, image retrieval systems, and automatic alt-text generation for social media.
"""
# Install required packages:
# pip install transformers torch torchvision pillow
 
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import matplotlib.pyplot as plt
 
# Load image from file
image_path = "dog_in_park.jpg"  # Replace with your image
raw_image = Image.open(image_path).convert('RGB')
 
# Load BLIP (Bootstrapped Language Image Pretraining) model and processor
# BLIP is a powerful model for vision-language tasks including captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
 
# Preprocess the image and prepare inputs for the model
inputs = processor(images=raw_image, return_tensors="pt")
 
# Generate caption
with torch.no_grad():
    output = model.generate(**inputs)
 
# Decode the generated token IDs to a human-readable string
caption = processor.decode(output[0], skip_special_tokens=True)
 
# Show image and caption
plt.imshow(raw_image)
plt.title(f"Caption: {caption}", fontsize=14)
plt.axis('off')
plt.show()