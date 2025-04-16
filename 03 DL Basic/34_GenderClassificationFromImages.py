"""
Project 114. Gender classification from images
Description:
Gender classification from images involves predicting whether a person is male or female based on facial features. In this project, we use a pre-trained deep learning model built on Caffe to detect a face and classify gender from a static image using OpenCV's DNN module.

About:
👩👨 What This Project Demonstrates:
Detects faces using deep learning-based OpenCV face detector

Predicts gender using a pre-trained classification model

Draws bounding boxes and prediction labels with confidence scores
"""

# Install if not already: pip install opencv-python opencv-contrib-python numpy
 
import cv2
import numpy as np
 
# Gender classes
GENDER_LIST = ["Male", "Female"]
 
# Load pre-trained gender classification model
gender_net = cv2.dnn.readNetFromCaffe(
    prototxt="https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/models/gender_deploy.prototxt",
    caffeModel="https://github.com/spmallick/learnopencv/raw/master/AgeGender/models/gender_net.caffemodel"
)
 
# Load pre-trained face detector model
face_net = cv2.dnn.readNetFromCaffe(
    prototxt="https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    caffeModel="https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)
 
# Load and preprocess image
image = cv2.imread(cv2.samples.findFile("messi5.jpg"))  # Replace with your image
(h, w) = image.shape[:2]
 
# Detect faces in the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), 
                             (104.0, 177.0, 123.0), swapRB=False, crop=False)
face_net.setInput(blob)
detections = face_net.forward()
 
# Loop over all detected faces
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.6:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
        # Extract the face ROI
        face = image[startY:endY, startX:endX]
        if face.size == 0:
            continue
        face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
                                          (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
 
        # Predict gender
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        confidence_score = gender_preds[0].max()
 
        # Draw bounding box and gender label
        label = f"Gender: {gender} ({confidence_score*100:.1f}%)"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
 
# Display the result
cv2.imshow("Gender Classification", image)
cv2.waitKey(0)
cv2.destroyAllWindows()