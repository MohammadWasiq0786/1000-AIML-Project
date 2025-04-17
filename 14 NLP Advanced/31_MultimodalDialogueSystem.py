"""
Project 551: Multimodal Dialogue System
Description:
A multimodal dialogue system incorporates multiple types of input data, such as text, images, or audio, to engage in more natural and dynamic interactions. In this project, we will create a simple multimodal dialogue system that combines text input and image data to generate responses based on both modalities. For instance, the system might respond to a user's query with both text and images.
"""

from transformers import pipeline
from PIL import Image
import requests
 
# 1. Load pre-trained models for text and image generation
text_generator = pipeline("text-generation", model="gpt2")
image_captioning = pipeline("image-captioning")
 
# 2. Function to generate a response based on both text and an image
def multimodal_response(user_input, image_path=None):
    # Text-based response generation
    text_response = text_generator(user_input, max_length=50)[0]['generated_text']
    
    # If an image is provided, generate a caption for it
    if image_path:
        image = Image.open(image_path)
        image_caption = image_captioning(image)
        response = f"Text Response: {text_response}\nImage Caption: {image_caption[0]['caption']}"
    else:
        response = f"Text Response: {text_response}"
    
    return response
 
# 3. Example user input and image (local path or URL to the image)
user_input = "Describe the picture of a cat."
image_path = "path_to_image_of_cat.jpg"  # Update with the actual image path
 
# 4. Generate multimodal response
response = multimodal_response(user_input, image_path)
print(response)