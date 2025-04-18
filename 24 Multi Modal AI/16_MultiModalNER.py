"""
Project 936. Multi-modal Named Entity Recognition (NER)

Multi-modal Named Entity Recognition (NER) extends traditional NER by recognizing entities not just from text, but from other modalities such as images or audio. For example, in a movie clip, we could extract entities like actors (from speech or subtitles) and objects (from the visual content of the video).

In this project, we simulate a basic multi-modal NER system that extracts entities from both text (using traditional NER) and images (using object detection).

Step 1: Text-based NER
We use spaCy for Named Entity Recognition on the textual content.

Step 2: Image-based NER
We use YOLO (You Only Look Once) or Haar Cascades to detect objects (acting as named entities) in images.

What This Does:
Text-based NER: Extracts named entities (like people, locations, and organizations) from the input text using spaCy's pre-trained NER model.

Image-based NER: Uses Haar Cascades (for simplicity) to detect objects in an image, such as faces.

Combines the results from both text and image into a multi-modal entity list.
"""

import spacy
import cv2
from PIL import Image
import numpy as np
 
# Step 1: Text-based NER using spaCy
nlp = spacy.load("en_core_web_sm")  # Load spaCy's pre-trained NER model
 
def extract_entities_from_text(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities
 
# Example text input
text_input = "Elon Musk is the CEO of SpaceX, and he lives in California."
text_entities = extract_entities_from_text(text_input)
print(f"Text-based NER: {text_entities}")
 
# Step 2: Image-based NER using OpenCV (Haar Cascades for simplicity)
def detect_objects_in_image(image_path):
    # Load pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    objects_detected = []
    for (x, y, w, h) in faces:
        objects_detected.append("Face")
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around the face
    
    # Save and display the image
    cv2.imwrite("output_image.jpg", img)
    Image.open("output_image.jpg").show()
    
    return objects_detected
 
# Example image input (replace with a valid image path)
image_path = "example_image.jpg"  # Replace with a valid image path
image_entities = detect_objects_in_image(image_path)
print(f"Image-based NER (Objects Detected): {image_entities}")
 
# Combining text and image NER results
final_entities = text_entities + [(entity, "Object") for entity in image_entities]
print(f"Combined Multi-modal NER: {final_entities}")