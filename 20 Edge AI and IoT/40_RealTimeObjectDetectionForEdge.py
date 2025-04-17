"""
Project 800: Real-Time Object Detection for Edge
Description
Real-time object detection on edge devices enables applications like surveillance, robotics, and smart retail without cloud dependency. We'll use a pre-trained SSD MobileNet model optimized with TensorFlow Lite for fast, low-power inference — suitable for devices like Raspberry Pi, Coral TPU, and NVIDIA Jetson.

✅ Ready for deployment on:

Raspberry Pi with PiCam + TFLite

Google Coral USB Accelerator

Jetson Nano using TensorRT conversion

Real-world use: real-time intruder detection, vehicle monitoring, or smart checkout systems
"""

import cv2
import numpy as np
import tensorflow as tf
 
# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="ssd_mobilenet_v1.tflite")  # You can use any downloaded TFLite object detector
interpreter.allocate_tensors()
 
# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
 
# Load label map (COCO 90 classes for MobileNet)
labels = {0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 5: 'bus', 7: 'truck'}
 
# Open webcam or video stream
cap = cv2.VideoCapture(0)  # use 0 for webcam, or replace with a video file path
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # Prepare frame
    input_shape = input_details[0]['shape']
    resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(resized, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5  # normalize as per SSD MobileNet requirement
 
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
 
    # Get detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
 
    # Draw results
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            (left, top, right, bottom) = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))
            label = labels.get(classes[i], "unknown")
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {scores[i]:.2f}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
 
    cv2.imshow('Edge Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()