"""
Project 125. YOLO implementation for object detection
Description:
YOLO (You Only Look Once) is a high-speed, real-time object detection system that detects multiple objects in a single pass of a convolutional neural network. In this project, we use a pre-trained YOLOv5 model to detect objects in an image and draw bounding boxes around them.

About:
📦 What This Project Demonstrates:
Loads a YOLOv5 model for real-time object detection

Detects and labels multiple objects with confidence scores

Draws bounding boxes over detected items in an image
"""

# Install if not already: pip install ultralytics opencv-python matplotlib
 
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
 
# Load pre-trained YOLOv5s model (small and fast)
model = YOLO("yolov5s.pt")  # Automatically downloads the model if not present
 
# Load input image
image_path = "test_image.jpg"  # Replace with your own image
image = cv2.imread(image_path)
 
# Run inference
results = model(image)
 
# Parse results and draw bounding boxes
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
 
        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
 
# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# Show image
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title("YOLOv5 Object Detection")
plt.axis("off")
plt.show()