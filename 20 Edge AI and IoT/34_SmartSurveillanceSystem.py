"""
Project 794: Smart Surveillance System
Description
A smart surveillance system uses computer vision to detect unusual activity, track people or objects, and issue alerts for security threats. AI enables real-time video stream analysis for applications like intrusion detection, loitering, or object abandonment. In this project, we simulate a basic person detection system using image input and a pre-trained object detection model (MobileNet-SSD).

✅ What this does:

Detects people in an image using MobileNet-SSD.

Can be extended to video streams using cv2.VideoCapture(0) for live feed.

Basis for further features: intrusion zones, abandoned object alerts, or time-based tracking.

For deployment, it can be installed on NVIDIA Jetson, Raspberry Pi (with Coral TPU), or smart IP cameras.
"""

import cv2
import numpy as np
 
# Load pre-trained MobileNet SSD model (for person detection)
net = cv2.dnn.readNetFromCaffe(
    'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/deploy.prototxt',
    'https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel'
)
 
# Class labels for COCO dataset (only 'person' needed here)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
 
# Load sample image (can also be a video frame)
image = cv2.imread("surveillance_sample.jpg")  # replace with your own image
(h, w) = image.shape[:2]
 
# Prepare image for detection
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()
 
# Loop over detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    idx = int(detections[0, 0, i, 1])
    
    # Only detect persons with high confidence
    if CLASSES[idx] == "person" and confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, f"Person: {confidence:.2f}", (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
# Display the image with detections
cv2.imshow("Smart Surveillance Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()