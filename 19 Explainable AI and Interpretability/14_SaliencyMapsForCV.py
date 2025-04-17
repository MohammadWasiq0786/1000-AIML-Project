"""
Project 734: Saliency Maps for Computer Vision
Description:
Saliency maps highlight the regions in an image that contribute the most to a model's prediction. These maps provide a way to visualize the importance of different parts of an image by showing which pixels the model focuses on when making a classification decision. Saliency maps are especially useful for interpreting Convolutional Neural Networks (CNNs) in computer vision tasks like image classification. In this project, we will implement saliency map generation for a CNN model and visualize which parts of an image influence the model’s predictions the most.

Explanation:
Image Preprocessing: The load_image() function loads and preprocesses an image to match the input requirements for VGG16 (224x224 pixels and normalized pixel values).

VGG16 Model: We load the pre-trained VGG16 model from Keras with ImageNet weights. This model is used for image classification tasks and is a commonly used CNN.

Saliency Map Generation: The generate_saliency_map() function calculates the saliency map by computing the gradients of the predicted class with respect to the input image. These gradients are then used to generate a heatmap, which highlights the important areas of the image for the model’s decision.

Visualization: The visualize_saliency_map() function overlays the saliency map on top of the original image, showing the regions that had the most influence on the model’s classification.

Saliency maps are particularly useful in model interpretability as they help understand which parts of an image the model is focusing on when making predictions.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
 
# 1. Load and preprocess the image
def load_image(img_path):
    """
    Load an image from the given path and preprocess it for VGG16.
    - Resizes the image to 224x224, converts to array, and normalizes it.
    """
    img = image.load_img(img_path, target_size=(224, 224))  # VGG16 expects 224x224 images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Preprocess for VGG16
    return img_array
 
# 2. Load pre-trained VGG16 model
def load_vgg16_model():
    """
    Load the pre-trained VGG16 model with ImageNet weights.
    """
    model = VGG16(weights='imagenet')
    return model
 
# 3. Generate saliency map
def generate_saliency_map(model, img_array, class_idx):
    """
    Generate a saliency map by calculating the gradient of the predicted class with respect to the input image.
    """
    # Compute the gradient of the output with respect to the input image
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        predictions = model(img_array)
        class_output = predictions[:, class_idx]  # Get the output for the target class
    
    # Calculate the gradients of the class output with respect to the input image
    grads = tape.gradient(class_output, img_array)
    
    # Get the absolute value of the gradients
    saliency_map = np.abs(grads[0].numpy())
    
    # Take the maximum value across the color channels (RGB)
    saliency_map = np.max(saliency_map, axis=-1)
    saliency_map = saliency_map / np.max(saliency_map)  # Normalize the saliency map
    
    return saliency_map
 
# 4. Visualize the saliency map
def visualize_saliency_map(saliency_map, img_path):
    """
    Visualize the saliency map by overlaying it on the original image.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
 
    # Plot the original image and saliency map
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_array / 255.0)
    plt.title("Original Image")
    plt.axis('off')
    
    # Saliency map overlay
    plt.subplot(1, 2, 2)
    plt.imshow(img_array / 255.0)
    plt.imshow(saliency_map, cmap='jet', alpha=0.6)  # Overlay the saliency map
    plt.title("Saliency Map")
    plt.axis('off')
    
    plt.show()
 
# 5. Example usage
img_path = 'path_to_image.jpg'  # Replace with the image path
img_array = load_image(img_path)
 
# Load the pre-trained model
model = load_vgg16_model()
 
# Get the model's predictions
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])  # Get the class with the highest predicted probability
 
# Generate the saliency map
saliency_map = generate_saliency_map(model, img_array, class_idx)
 
# Visualize the saliency map
visualize_saliency_map(saliency_map, img_path)