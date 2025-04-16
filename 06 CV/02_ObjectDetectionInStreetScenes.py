"""
Project 202. Object detection in street scenes
Description:
Object Detection identifies and localizes multiple objects within an image using bounding boxes and class labels. In this project, we use a pretrained YOLOv5 model to detect vehicles, pedestrians, traffic signs, and more in urban street scenes, making it ideal for self-driving cars, surveillance, and smart city applications.

About:
🧠 What This Project Demonstrates:
Performs real-time object detection on complex street scenes

Identifies multiple classes like cars, people, bikes, trucks

Uses YOLOv8 – a fast, state-of-the-art object detection model
"""

# Install if not already:
# pip install ultralytics
# Requires: Python >= 3.8, Torch >= 1.7
 
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
 
# Load pretrained YOLOv5 model (small, fast, accurate)
model = YOLO('yolov8n.pt')  # Or 'yolov8s.pt' for more accuracy
 
# Load a sample street image
image_path = 'street_scene.jpg'  # Replace with your own image path
results = model(image_path)
 
# Show detection results
results.print()
results.show()  # Displays image with bounding boxes in a pop-up window
 
# Optional: Save the result image
results.save(save_dir='detections/')  # Saves image with boxes to folder
# If you don’t have an image, you can test it with a sample:

import requests
from PIL import Image
 
# Download a street scene image
url = 'https://images.pexels.com/photos/167832/pexels-photo-167832.jpeg'
img_path = 'street_scene.jpg'
img_data = requests.get(url).content
with open(img_path, 'wb') as f:
    f.write(img_data)
 
# Then rerun the detection
results = model(img_path)
results.show()