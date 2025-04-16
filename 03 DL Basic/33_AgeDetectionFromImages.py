"""
Project 113. Age detection from images
Description:
Age detection from images aims to estimate a person’s age based on facial features. In this project, we use a pre-trained deep learning model to predict the approximate age group from a face image. This is often used in demographics analysis, targeted advertising, or access control systems.

About:
👶🧓 What This Project Demonstrates:
Detects faces using OpenCV's DNN face detector

Estimates age using a pre-trained CNN model (Caffe)

Works on single or multiple faces in an image
"""

# Install if not already: pip install opencv-python opencv-contrib-python numpy
 
import cv2
import numpy as np
 
# Load the age categories used by the model
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
 
# Load pre-trained Caffe models
age_model = cv2.dnn.readNetFromCaffe(
    prototxt="https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/models/age_deploy.prototxt",
    caffeModel="https://github.com/spmallick/learnopencv/raw/master/AgeGender/models/age_net.caffemodel"
)
 
face_model = cv2.dnn.readNetFromCaffe(
    prototxt="https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    caffeModel="https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)
 
# Load input image
image = cv2.imread(cv2.samples.findFile("messi5.jpg"))  # Replace with your image
(h, w) = image.shape[:2]
 
# Detect face
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
face_model.setInput(blob)
detections = face_model.forward()
 
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.6:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
        # Extract face ROI
        face = image[startY:endY, startX:endX]
        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
                                          (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
 
        # Predict age
        age_model.setInput(face_blob)
        age_preds = age_model.forward()
        age = AGE_BUCKETS[age_preds[0].argmax()]
        confidence_score = age_preds[0].max()
 
        # Draw the result
        text = f"Age: {age} ({confidence_score*100:.1f}%)"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, text, (startX, startY - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
 
# Show output
cv2.imshow("Age Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()