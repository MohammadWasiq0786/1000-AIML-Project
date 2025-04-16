"""
Project 96. Object detection in images
Description:
Object detection involves identifying and localizing objects in images using bounding boxes and class labels. In this project, we use a pre-trained YOLOv5 or TensorFlow’s SSD MobileNet model to perform object detection on input images with minimal code and setup.

About:
🧠 What This Project Demonstrates:
Loads a pre-trained object detection model (SSD MobileNet)

Performs bounding box localization and classification

Visualizes detected objects using OpenCV and Matplotlib
"""

# Install if not already: pip install tensorflow tensorflow-hub opencv-python matplotlib
 
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
# Load a pre-trained SSD MobileNet model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
 
# Load and preprocess image
def load_image(path):
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(img_rgb, (320, 320))
    return img, tf.expand_dims(resized / 255.0, axis=0)
 
# Draw bounding boxes
def draw_boxes(image, boxes, class_names, scores, threshold=0.5):
    height, width, _ = image.shape
    for i in range(min(len(boxes), 20)):
        if scores[i] >= threshold:
            y1, x1, y2, x2 = boxes[i]
            start_point = (int(x1 * width), int(y1 * height))
            end_point = (int(x2 * width), int(y2 * height))
            label = class_names[i].decode("ascii")
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {scores[i]:.2f}", (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image
 
# Load labels
labels_path = tf.keras.utils.get_file(
    'mscoco_labels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/mscoco_label_map.txt'
)
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()
 
# Image path (use any .jpg/.png image)
image_path = 'example.jpg'  # Replace with your image path
original_img, input_tensor = load_image(image_path)
 
# Run inference
result = detector(input_tensor)
result = {key: value.numpy() for key, value in result.items()}
 
# Draw and display
output_img = draw_boxes(original_img.copy(), result['detection_boxes'],
                        [labels[int(i)] for i in result['detection_classes']],
                        result['detection_scores'])
 
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title("Object Detection with SSD MobileNet")
plt.axis("off")
plt.tight_layout()
plt.show()