"""
Project 726: Grad-CAM Implementation
Description:
Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique for visualizing the regions of an image that contribute the most to a deep learning model's prediction. It uses the gradients of the target class with respect to the final convolutional layer to produce a heatmap. The heatmap highlights which regions of the image the model focuses on when making a prediction, which is useful for model interpretability.

About:
Explanation:
Image Preprocessing: The load_image() function loads and preprocesses the image for ResNet50 by resizing it to 224x224 and normalizing the pixel values to match the model’s expected input.

Grad-CAM Generation: The generate_grad_cam() function computes the Grad-CAM by calculating the gradients of the predicted class with respect to the last convolutional layer. The gradients are then used to create a heatmap that highlights important regions in the image.

Overlay Heatmap: The overlay_heatmap_on_image() function superimposes the Grad-CAM heatmap on top of the original image, allowing us to visualize which regions are most important for the model's decision-making process.

Visualize: The generated heatmap is visualized using matplotlib, showing which areas of the image the model focuses on when making the classification.

This method is very useful for model interpretability, as it allows us to see what parts of an image are driving the model’s predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
 
# 1. Load and preprocess the image
def load_image(img_path):
    """
    Load an image from the given path and preprocess it for ResNet50.
    - Resizes the image to 224x224, converts to array, and normalizes it.
    """
    img = image.load_img(img_path, target_size=(224, 224))  # ResNet50 expects 224x224 images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array
 
# 2. Load pre-trained ResNet50 model
def load_resnet50_model():
    """
    Load the pre-trained ResNet50 model with ImageNet weights.
    """
    model = ResNet50(weights='imagenet')
    return model
 
# 3. Generate Grad-CAM (Class Activation Map)
def generate_grad_cam(model, img_array, class_idx):
    """
    Generate Grad-CAM for the given image and class index.
    - Uses the gradients of the predicted class with respect to the last conv layer.
    """
    # Get the model's prediction and the output layer
    last_conv_layer = model.get_layer('conv5_block3_out')  # Last convolutional layer
    model_out = model.output[:, class_idx]  # Output for the target class
    
    # Compute the gradients of the predicted class with respect to the last conv layer's output
    grads = tf.gradients(model_out, last_conv_layer.output)[0]
    grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)  # Normalize gradients
    
    # Get the feature map of the last convolutional layer
    conv_layer_output = last_conv_layer.output[0]
    
    # Multiply each channel in the feature map by the corresponding gradient
    heatmap = np.dot(conv_layer_output, grads.numpy())
    heatmap = np.maximum(heatmap, 0)  # Apply ReLU
    heatmap = heatmap / np.max(heatmap)  # Normalize the heatmap
    
    return heatmap
 
# 4. Superimpose Grad-CAM heatmap on the original image
def overlay_heatmap_on_image(heatmap, img_path):
    """
    Overlay the Grad-CAM heatmap on the original image to highlight important regions.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    heatmap_resized = np.uint8(np.clip(255 * heatmap, 0, 255))  # Normalize heatmap
    
    # Create the heatmap image with a colormap
    plt.imshow(img_array / 255.0)  # Original image
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.6)  # Heatmap overlay
    plt.axis('off')  # Hide axes
    plt.show()
 
# 5. Example usage
img_path = 'path_to_image.jpg'  # Replace with the image path
img_array = load_image(img_path)
 
# Load the model
model = load_resnet50_model()
 
# Get the model's predictions
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])  # Get the class with the highest predicted probability
 
# Generate Grad-CAM heatmap
heatmap = generate_grad_cam(model, img_array, class_idx)
 
# Overlay the heatmap on the image
overlay_heatmap_on_image(heatmap, img_path)