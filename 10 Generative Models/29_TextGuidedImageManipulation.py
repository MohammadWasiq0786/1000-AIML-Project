"""
Project 389. Text-guided image manipulation
Description:
Text-guided image manipulation involves changing an existing image based on a textual description. This technique can be used to modify specific aspects of an image, such as changing the color, background, or adding objects, based on a given prompt. For example, one could ask the model to “make the sky bluer” or “add a cat to the image,” and the model would modify the image accordingly. In this project, we will explore how to use generative models like GANs or Diffusion Models in combination with text inputs to manipulate images.

About:
✅ What It Does:
CLIP is used to evaluate the relationship between the text input and the image by calculating the similarity scores.

This project simulates text-guided manipulation by determining how well an image matches a given text description.

The similarity scores can be used to guide image transformation models in more complex systems for manipulating images based on text.

Key features:
CLIP helps to align images with text descriptions, making it possible to guide modifications to images based on textual input.

Although this code does not perform actual image manipulation, it serves as a basis for integrating with more advanced models (e.g., GANs or Diffusion Models) to modify the image accordingly.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
 
# 1. Load pre-trained CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)
 
# 2. Define the text input and the image to manipulate
text_input = "Add a bright sunset in the background"
image_path = "path_to_image.jpg"  # Provide the path to your image here
 
# 3. Load and process the image
image = Image.open(image_path)
 
# 4. Process the text and image for CLIP model
inputs = clip_processor(text=text_input, images=image, return_tensors="pt", padding=True)
 
# 5. Get the model's predictions for the text-image pair
outputs = clip_model(**inputs)
logits_per_image = outputs.logits_per_image  # Similarity between text and image
logits_per_text = outputs.logits_per_text  # Similarity between image and text
 
# 6. Display the similarity scores (you can tune this part for more advanced manipulation)
print(f"Text-Image Similarity: {logits_per_image.item()}")
print(f"Image-Text Similarity: {logits_per_text.item()}")
 
# 7. Display the image (No manipulation here, but this is where you would adjust the image)
plt.imshow(image)
plt.title(f"Text: {text_input}")
plt.show()
 
# For more advanced manipulation, you'd typically feed these outputs to a GAN or another model
# that adjusts the image based on the text input.