"""
Project 110. Super-resolution model
Description:
Super-resolution refers to enhancing the resolution of an image, turning a low-resolution (LR) input into a high-resolution (HR) output. In this project, we use a pre-trained EDSR (Enhanced Deep Super-Resolution) model from TensorFlow Hub to upscale images while preserving fine details.

🔍 What This Project Demonstrates:
Upscales low-res images using a deep neural network

Leverages a pre-trained EDSR/ESRGAN model from TensorFlow Hub

Retains fine textures and sharp edges
"""


# Install if not already: pip install tensorflow tensorflow-hub matplotlib pillow
 
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
 
# Load the pre-trained EDSR model (4x super-resolution)
model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
 
# Load low-resolution image
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((100, 100))  # Simulate low-res image
    return np.array(img) / 255.0  # Normalize to [0, 1]
 
# Preprocess and convert to tensor
def preprocess(img):
    lr = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), 0)
    return lr
 
# Postprocess and convert back to image
def postprocess(sr):
    sr_img = tf.squeeze(sr).numpy()
    sr_img = np.clip(sr_img * 255, 0, 255).astype(np.uint8)
    return sr_img
 
# Load and enhance image
image_path = tf.keras.utils.get_file('lowres.jpg', 'https://i.imgur.com/F28w3Ac.jpg')
low_res = load_image(image_path)
lr_tensor = preprocess(low_res)
 
# Super-resolution inference
sr_tensor = model(lr_tensor)
sr_image = postprocess(sr_tensor)
 
# Display results
plt.figure(figsize=(12, 5))
 
plt.subplot(1, 2, 1)
plt.title("Low-Resolution Input")
plt.imshow(low_res)
plt.axis("off")
 
plt.subplot(1, 2, 2)
plt.title("Super-Resolved Output")
plt.imshow(sr_image)
plt.axis("off")
 
plt.tight_layout()
plt.show()