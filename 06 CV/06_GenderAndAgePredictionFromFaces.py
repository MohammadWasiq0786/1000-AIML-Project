"""
Project 206. Gender and age prediction from faces
Description:
Gender and Age Prediction estimates a person’s gender and approximate age from a facial image. It’s widely used in retail analytics, smart mirrors, demographic studies, and personalized user interfaces. This project uses a pretrained deep learning model based on OpenCV DNN to perform real-time age and gender detection.

About:
🧠 What This Project Demonstrates:
Predicts gender (male/female) and age range using pretrained deep models

Uses OpenCV DNN module with Caffe-based models

Can be easily extended to video streams or webcam input
"""

# Install if not already: pip install opencv-python
 
import cv2
 
# Load pretrained model files (download from OpenCV or GitHub repositories)
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
 
# Load the models
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
 
# Age and gender labels
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
 
# Load a face image
image_path = 'face_sample.jpg'
image = cv2.imread(image_path)
(h, w) = image.shape[:2]
face_blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), 
                                  (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
 
# Predict gender
gender_net.setInput(face_blob)
gender_preds = gender_net.forward()
gender = GENDER_LIST[gender_preds[0].argmax()]
 
# Predict age
age_net.setInput(face_blob)
age_preds = age_net.forward()
age = AGE_BUCKETS[age_preds[0].argmax()]
 
# Display results
label = f"{gender}, {age}"
cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Age & Gender Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()