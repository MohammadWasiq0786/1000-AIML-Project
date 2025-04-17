"""
Project 743: Interpretable Computer Vision Models
Description:
Interpretable computer vision models aim to make deep learning models used in image classification and other vision tasks more transparent. These models are often viewed as "black-box" systems, where the decision-making process is difficult to understand. By using techniques such as saliency maps, Grad-CAM, and LIME, we can visualize which parts of an image influence the model's predictions. In this project, we will train a Convolutional Neural Network (CNN) on an image dataset and apply techniques like Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the areas of the image that contribute most to the model's predictions.

Explanation:
Dataset Loading and Preprocessing: We load the CIFAR-10 dataset, which contains 60,000 images in 10 classes. We normalize the pixel values to the range [0, 1] for easier training.

CNN Model: A simple Convolutional Neural Network (CNN) model is built for image classification. It has two convolutional layers followed by a dense layer for classification into 10 classes.

Grad-CAM Generation: The generate_grad_cam() function computes the Grad-CAM heatmap. This function uses the last convolutional layer's gradients with respect to the class of interest to generate a heatmap that highlights the important regions of the image.

Visualization: The display_grad_cam() function overlays the Grad-CAM heatmap on top of the input image to visualize which regions of the image contribute most to the model’s decision.

Example Usage: We train the CNN on CIFAR-10 and use Grad-CAM to interpret the model's decision-making for a selected image. The heatmap shows which regions of the image (e.g., the presence of certain objects) are most important for the model’s classification.

By using Grad-CAM, we can make CNN models more interpretable, allowing us to understand which parts of the image are driving the model's decisions.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
 
# 1. Load and preprocess the CIFAR-10 dataset
def load_and_preprocess_data():
    """
    Load the CIFAR-10 dataset and preprocess it.
    CIFAR-10 consists of 60,000 32x32 color images in 10 classes.
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Normalize the images to [0, 1]
    X_train, X_test = X_train / 255.0, X_test / 255.0
    return X_train, X_test, y_train, y_test
 
# 2. Build a simple CNN model
def build_cnn_model(input_shape):
    """
    Build a simple CNN model for image classification.
    """
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # 10 output classes for CIFAR-10
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 3. Grad-CAM: Compute Gradients and Heatmap
def generate_grad_cam(model, image, class_idx):
    """
    Generate a Grad-CAM heatmap for the given image and class index.
    """
    # Get the model's output layer (last convolutional layer)
    last_conv_layer = model.get_layer('conv2d_2')
    
    # Create a model that gives the gradient of the class output with respect to the last conv layer output
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        loss = predictions[:, class_idx]
    
    # Compute the gradients
    grads = tape.gradient(loss, conv_output)
    
    # Pool the gradients across all the filters
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    # Multiply the pooled gradients with the convolutional layer output
    conv_output = conv_output[0]
    heatmap = np.dot(conv_output, pooled_grads)
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap, conv_output
 
# 4. Visualize Grad-CAM heatmap
def display_grad_cam(image, heatmap):
    """
    Display the Grad-CAM heatmap on the original image.
    """
    # Resize heatmap to match the input image size
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    
    # Convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.dstack([heatmap, np.zeros_like(heatmap), 255-heatmap])
    
    # Superimpose the heatmap on the image
    superimposed_img = np.array(image[0])
    superimposed_img = cv2.addWeighted(superimposed_img, 0.6, heatmap, 0.4, 0)
    
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()
 
# 5. Example usage
X_train, X_test, y_train, y_test = load_and_preprocess_data()
 
# Build the CNN model
model = build_cnn_model(input_shape=(32, 32, 3))
 
# Train the model on CIFAR-10
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
 
# Choose an image from the test set to visualize Grad-CAM
img_idx = 10  # Example index
image_to_explain = X_test[img_idx:img_idx+1]  # Shape: (1, 32, 32, 3)
 
# Make a prediction to get the class index
predictions = model.predict(image_to_explain)
class_idx = np.argmax(predictions[0])
 
# Generate Grad-CAM heatmap
heatmap, _ = generate_grad_cam(model, image_to_explain, class_idx)
 
# Visualize Grad-CAM on the image
display_grad_cam(image_to_explain, heatmap)