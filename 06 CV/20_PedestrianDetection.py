"""
Project 220. Pedestrian detection
Description:
Pedestrian detection focuses on identifying and locating people in images or video frames. It’s a critical component in autonomous driving, surveillance systems, and smart city applications. In this project, we’ll use a pre-trained object detection model like YOLOv5 or Faster R-CNN to detect pedestrians in an image.

About:
What It Does:
This project detects people in a visual scene, ideal for real-time systems like collision avoidance, crowd monitoring, and traffic analysis. You can enhance it further with tracking, thermal vision, or integration into video feeds for live pedestrian analytics.
"""
# Install YOLOv5 dependencies:
# pip install ultralytics opencv-python matplotlib
 
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
 
# Load the YOLOv5 model (small version is fast and efficient)
model = YOLO("yolov5s.pt")  # This downloads the model if not present
 
# Load the input image
image_path = "crosswalk.jpg"  # Replace with your image
image = cv2.imread(image_path)
 
# Convert BGR to RGB for processing
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# Run inference on the image
results = model(rgb_image)[0]  # YOLO returns results in a structured format
 
# Draw bounding boxes for only 'person' class (class ID 0 in COCO)
for det in results.boxes:
    class_id = int(det.cls)
    if class_id == 0:  # Class 0 corresponds to 'person'
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        confidence = float(det.conf)
        label = f"Person {confidence:.2f}"
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
# Show result using matplotlib
plt.figure(figsize=(12, 6))
plt.imshow(rgb_image)
plt.title("Pedestrian Detection")
plt.axis('off')
plt.show()