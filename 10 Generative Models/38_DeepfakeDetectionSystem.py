"""
Project 398. Deepfake detection system
Description:
Deepfakes are hyper-realistic but manipulated videos or images created using generative models like GANs. They can be used for various purposes, including entertainment, but also for misinformation. A deepfake detection system aims to identify these manipulated media by examining the inconsistencies in the video or image, such as unusual facial expressions, unnatural lighting, or inconsistencies in the background.

In this project, we will implement a deepfake detection system using a pre-trained deep learning model to classify whether a video or image is a deepfake or real.

About:
✅ What It Does:
Deepfake Detection uses a pre-trained Xception model to classify images (or frames from a video) as real or fake.

The model processes the input image, and the CNN-based architecture is used to detect discrepancies typical of deepfakes.

Visualizes the input image and provides a prediction based on the model.

Key features:
Xception model is used here for image classification, but other models like EfficientNet or ResNet can also be used for deepfake detection.

This model can be fine-tuned with a deepfake dataset like FaceForensics++ for better results.

The real/fake prediction can be used for video deepfake detection by processing frames individually.
"""

# pip install tensorflow keras opencv-python

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
# 1. Load the pre-trained Xception model for deepfake detection
model = Xception(weights='imagenet', include_top=True)
 
# 2. Load and preprocess the input image (replace 'image_path' with your image path)
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_resized = cv2.resize(img, (299, 299))  # Resize to 299x299 (input size for Xception)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input(img_array)
 
# 3. Make predictions using the model
def predict_deepfake(image_path):
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)
    return prediction
 
# 4. Example image path
image_path = "deepfake_image.jpg"  # Path to the image to be tested (deepfake or real)
 
# 5. Get the prediction
prediction = predict_deepfake(image_path)
 
# 6. Display the result
print(f"Prediction: {prediction}")
 
# 7. Visualize the image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Input Image")
plt.show()