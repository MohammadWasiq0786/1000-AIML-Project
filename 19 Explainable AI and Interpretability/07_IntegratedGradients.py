"""
Project 727: Integrated Gradients Implementation
Description:
Integrated Gradients is a method for explaining the predictions of machine learning models by attributing the prediction to individual features. It works by calculating the integral of the gradients of the model's output with respect to the input features along a straight line from a baseline (e.g., a black image or a zero-input image) to the actual input. The accumulated gradients along this path are used to assign importance to each feature. This method helps in understanding how changes in the input features affect the model's output.

Explanation:
Image Preprocessing: The load_image() function loads and preprocesses an image, resizing it to 224x224 pixels and normalizing it for the ResNet50 model.

Model Loading: We use ResNet50, a pre-trained CNN model, to perform image classification on the input image.

Compute Integrated Gradients: The compute_integrated_gradients() function calculates the integrated gradients by performing linear interpolation between a baseline image (usually a black image or zero input) and the actual image, then calculating gradients at each step and averaging them. This gives us the attribution of features that contributed to the predicted class.

Visualization: The visualize_integrated_gradients() function visualizes the integrated gradients by overlaying them on the original image. The heatmap generated shows which parts of the image contributed most to the prediction.

This approach gives us insight into which features of the image were most important for the model’s decision, enhancing the explainability of the model.
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
 
# 3. Compute gradients and integrated gradients
def compute_integrated_gradients(model, img_array, class_idx, baseline=None, steps=50):
    """
    Calculate Integrated Gradients by computing gradients along the path from baseline to input image.
    """
    if baseline is None:
        baseline = np.zeros(img_array.shape)  # Use a black image as baseline
    
    # Compute the gradients for the model's output
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        predictions = model(img_array)
    
    grads = tape.gradient(predictions[:, class_idx], img_array)
    
    # Linearly interpolate between baseline and input image
    alpha = np.linspace(0, 1, steps)
    integrated_grads = np.zeros_like(img_array)
    for i in alpha:
        interpolated_image = baseline + i * (img_array - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated_image)
            preds = model(interpolated_image)
        grads_at_interpolation = tape.gradient(preds[:, class_idx], interpolated_image)
        integrated_grads += grads_at_interpolation / steps
    
    return integrated_grads[0]
 
# 4. Visualize the integrated gradients
def visualize_integrated_gradients(integrated_grads, img_path):
    """
    Visualize the integrated gradients by overlaying them on the original image.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
 
    # Normalize the integrated gradients
    integrated_grads = np.abs(integrated_grads)
    integrated_grads /= np.max(integrated_grads)  # Normalize
    
    # Plot the original image and integrated gradients
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array / 255.0)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(integrated_grads, cmap='jet', alpha=0.7)  # Heatmap of gradients
    plt.title("Integrated Gradients")
    plt.axis('off')
    
    plt.show()
 
# 5. Example usage
img_path = 'path_to_image.jpg'  # Replace with the image path
img_array = load_image(img_path)
 
# Load the pre-trained model
model = load_resnet50_model()
 
# Get the model's predictions
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])  # Get the class with the highest predicted probability
 
# Compute the integrated gradients
integrated_grads = compute_integrated_gradients(model, img_array, class_idx)
 
# Visualize the integrated gradients
visualize_integrated_gradients(integrated_grads, img_path)