"""
Project 212. Visual question answering
Description:
Visual Question Answering is a multimodal AI task where a system is given an image and a natural language question, and it must generate a relevant answer based on the visual content. It combines computer vision (to interpret the image) and NLP (to understand the question). We'll implement it using the pre-trained BLIP model from Hugging Face, which is capable of VQA out-of-the-box.

About:
What It Does:
This project creates a system where you can ask questions about any image, and the AI will respond appropriately — e.g., "How many people are in the photo?" or "Is it daytime?" It's used in AI assistants, accessibility tools, education, and smart surveillance systems.
"""
# Install necessary packages first:
# pip install transformers torch torchvision pillow
 
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
import matplotlib.pyplot as plt
 
# Load the image
image_path = "zebra_photo.jpg"  # Replace with your image
image = Image.open(image_path).convert('RGB')
 
# Define your question related to the image
question = "What animal is in the picture?"
 
# Load the BLIP model and processor for Visual Question Answering
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
 
# Preprocess image and question for the model
inputs = processor(image, question, return_tensors="pt")
 
# Generate answer from the model
with torch.no_grad():
    output = model.generate(**inputs)
 
# Decode the predicted answer tokens
answer = processor.decode(output[0], skip_special_tokens=True)
 
# Display the image with the question and answer
plt.imshow(image)
plt.title(f"Q: {question}\nA: {answer}", fontsize=14)
plt.axis('off')
plt.show()