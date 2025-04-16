"""
Project 393. Sketch generation models
Description:
Sketch generation is the task of converting a real image into a sketch or line drawing that approximates the content of the original image. This process is useful for artistic applications, image-to-sketch conversion, and style transfer tasks. Generative models like GANs or Autoencoders can be used to learn the mapping from real images to their corresponding sketches. In this project, we will explore how to generate sketches from images using a pre-trained model.

About:
✅ What It Does:
Generates a sketch from a real image by passing it through a pre-trained neural network (e.g., trained on a dataset of images and sketches).

Uses image preprocessing (resizing, normalization) to match the model's input requirements.

Displays both the original image and its generated sketch side by side for comparison.

Key features:
Image-to-sketch conversion can be applied to any real image, converting it into an artistic sketch.

This implementation assumes the availability of a pre-trained sketch generation model, which can be replaced with other models like pix2pix or CycleGAN for more advanced results.
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
# 1. Load pre-trained model (Using a simple edge detection model or a model trained on sketch generation)
model = tf.keras.models.load_model('path_to_pretrained_sketch_model.h5')  # This is a placeholder
 
# 2. Function to generate sketch from an image
def generate_sketch(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
 
    # Resize the image to the input size expected by the model
    img_resized = cv2.resize(img, (256, 256))
    img_normalized = img_resized / 255.0  # Normalize image
 
    # Expand dimensions for the model
    img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
 
    # Predict the sketch using the model
    sketch = model.predict(img_input)[0]
    sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)  # Convert back to RGB
    return sketch
 
# 3. Path to the image to convert into a sketch
image_path = "path_to_your_image.jpg"
 
# 4. Generate and display the sketch
sketch_image = generate_sketch(image_path)
 
# 5. Display original image and generated sketch
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')
 
axes[1].imshow(sketch_image)
axes[1].set_title('Generated Sketch')
axes[1].axis('off')
 
plt.show()