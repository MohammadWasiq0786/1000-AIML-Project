"""
Project 796: Edge Vision Applications
Description
Edge vision applications use local AI inference to analyze visual input (images or video) directly on edge devices — reducing latency, bandwidth usage, and protecting privacy. In this project, we simulate a real-time object classification app using a lightweight MobileNetV2 model, suitable for deployment on edge devices like Raspberry Pi or Jetson Nano.

✅ Edge deployment ready:

Convert this model using TensorFlow Lite for fast inference on devices like:

Raspberry Pi with Coral TPU

Jetson Nano

ESP32-CAM with lightweight custom CNNs
"""

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
 
# Load the lightweight MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet')
 
# Load or capture image (simulated frame from camera)
frame = cv2.imread("sample_image.jpg")  # Replace with actual camera frame
 
# Preprocess frame for MobileNetV2
img = cv2.resize(frame, (224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)  # Normalize as expected by MobileNetV2
 
# Perform prediction
preds = model.predict(img_array)
top_preds = decode_predictions(preds, top=3)[0]
 
# Display results
for i, (imagenetID, label, prob) in enumerate(top_preds):
    print(f"Prediction {i+1}: {label} ({prob*100:.2f}%)")
 
# Overlay prediction on image
label_text = f"{top_preds[0][1]}: {top_preds[0][2]*100:.2f}%"
cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Edge Vision Classification", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()