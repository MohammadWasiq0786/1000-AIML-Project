"""
Project 112. Neural style transfer
Description:
Neural Style Transfer is a technique that blends the content of one image with the style of another using deep neural networks. It extracts high-level content representations from a content image and low-level texture/style representations from a style image, then synthesizes a new image that combines both. In this project, we use TensorFlow’s built-in tools for a fast and effective implementation.

About:
🎨 What This Project Demonstrates:
Uses deep learning to combine the style and content of two images

Employs a pre-trained arbitrary style transfer model from TensorFlow Hub

Produces stunning AI-generated art in seconds
"""

# Install if not already: pip install tensorflow tensorflow_hub matplotlib pillow
 
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
 
# Load the pretrained style transfer model from TF Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
 
# Function to load and preprocess image
def load_img(url, target_size=(256, 256)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return tf.constant(img, dtype=tf.float32)[tf.newaxis, ...]
 
# URLs for images (replace with your own)
content_url = "https://tensorflow.org/images/content.jpg"
style_url = "https://tensorflow.org/images/style.jpg"
 
# Load content and style images
content_image = load_img(content_url)
style_image = load_img(style_url)
 
# Apply style transfer
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
 
# Plot results
plt.figure(figsize=(12, 4))
 
plt.subplot(1, 3, 1)
plt.title("🎨 Style Image")
plt.imshow(style_image[0])
plt.axis("off")
 
plt.subplot(1, 3, 2)
plt.title("📸 Content Image")
plt.imshow(content_image[0])
plt.axis("off")
 
plt.subplot(1, 3, 3)
plt.title("🖼️ Stylized Image")
plt.imshow(stylized_image[0])
plt.axis("off")
 
plt.tight_layout()
plt.show()