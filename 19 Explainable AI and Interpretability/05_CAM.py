"""
Project 725: Class Activation Mapping (CAM)
Description:
Class Activation Mapping (CAM) is a technique used to highlight the regions of an input image that are important for a convolutional neural network's (CNN) decision-making process. CAM provides a visualization of where the model "looks" in the image to make a particular classification. This is particularly useful for image classification models, where we want to understand the spatial regions of the image that most influence the predicted class. In this project, we will implement Class Activation Mapping (CAM) to visualize the regions in an image that contribute to the model’s prediction.

For this project, we will use a pre-trained CNN model (like ResNet) and implement CAM to visualize the important regions for image classification. We will use the Grad-CAM method, a popular variant of CAM, which computes the gradients of the target class with respect to the final convolutional layer and uses them to generate the class activation map.

Required Libraries:
pip install tensorflow matplotlib numpy

Explanation:
Image Preprocessing: We load and preprocess the input image to match the input size and preprocessing requirements of ResNet50. The image is resized to 224x224 pixels and normalized using the preprocessing function for ResNet50.

Grad-CAM Generation: We compute the Grad-CAM by calculating the gradients of the predicted class with respect to the last convolutional layer. These gradients are then used to weigh the channels in the convolutional layer’s output, and we generate a heatmap showing which regions of the image are most important for the prediction.

Overlay Heatmap on Image: The generated Grad-CAM heatmap is overlaid on the original image, highlighting the regions that the model focuses on for the specific class prediction.

Class Activation Mapping Visualization: The heatmap is visualized using matplotlib, where the heatmap is shown on top of the original image, with higher intensity (red regions) representing areas of greater importance.

This technique allows us to visually understand what regions in an image are contributing most to the model's decision-making process, making the model's predictions more interpretable.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
 
# 1. Load and preprocess the image
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # ResNet50 expects 224x224 images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array
 
# 2. Load pre-trained ResNet50 model
def load_resnet50_model():
    model = ResNet50(weights='imagenet')
    return model
 
# 3. Generate Grad-CAM (Class Activation Map)
def generate_grad_cam(model, img_array, class_idx):
    # Get the model's prediction and the output layer
    last_conv_layer = model.get_layer('conv5_block3_out')  # Last convolutional layer
    model_out = model.output[:, class_idx]  # Output for the target class
    
    # Compute the gradients of the predicted class with respect to the last conv layer's output
    grads = tf.gradients(model_out, last_conv_layer.output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling
    
    # Get the feature map of the last convolutional layer
    conv_layer_output = last_conv_layer.output[0]
    
    # Multiply each channel in the feature map by the corresponding gradient
    heatmap = np.dot(conv_layer_output, pooled_grads.numpy())
    heatmap = np.maximum(heatmap, 0)  # Apply ReLU
    heatmap = heatmap / np.max(heatmap)  # Normalize the heatmap
    
    return heatmap
 
# 4. Superimpose Grad-CAM heatmap on the original image
def overlay_heatmap_on_image(heatmap, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    heatmap_resized = np.uint8(np.clip(255 * heatmap, 0, 255))
    heatmap_resized = np.expand_dims(heatmap_resized, axis=-1)  # Add channel dimension
    
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