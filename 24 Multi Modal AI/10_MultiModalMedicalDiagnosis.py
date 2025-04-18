"""
Project 930. Multi-modal Medical Diagnosis

Multi-modal medical diagnosis systems combine medical imaging (e.g., X-rays, MRIs) with patient data (e.g., medical history, lab results) to improve diagnostic accuracy. In this project, we simulate a system that uses both radiology images and textual patient information to predict potential medical conditions.

What This Does:
CLIP is used to process both the medical images (X-rays) and textual descriptions (diagnosis) of the medical conditions.

It calculates the cosine similarity between the patient's symptoms and the available medical diagnoses
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
 
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulated medical image dataset and corresponding diagnosis text
medical_data = [
    {"image": "xray_image1.jpg", "text": "Chest X-ray showing signs of pneumonia."},
    {"image": "xray_image2.jpg", "text": "X-ray indicating a healthy lung."},
    {"image": "xray_image3.jpg", "text": "Chest X-ray showing signs of tuberculosis."},
    {"image": "xray_image4.jpg", "text": "X-ray showing a clear heart and lung fields."}
]
 
# Simulated patient information (e.g., patient symptoms, diagnosis history)
patient_info = "Patient is coughing persistently and experiencing chest pain."
 
# Function to predict the most relevant diagnosis based on the patient's info and X-ray image
def predict_medical_condition(patient_info, medical_data, top_n=2):
    # Process the medical images and associated diagnosis texts
    inputs = processor(text=[item['text'] for item in medical_data], 
                       images=[Image.open(item['image']) for item in medical_data], 
                       return_tensors="pt", padding=True)
 
    # Process the patient's symptoms
    patient_input = processor(text=[patient_info] * len(medical_data), 
                              images=[Image.open(item['image']) for item in medical_data], 
                              return_tensors="pt", padding=True)
 
    # Get model outputs for the images and texts
    outputs = model(**inputs)
    patient_outputs = model(**patient_input)
 
    # Calculate cosine similarity between the patient's input and the diagnosis results
    similarity_scores = torch.cosine_similarity(outputs.text_embeds, patient_outputs.text_embeds)
 
    # Rank products based on similarity scores
    scores = similarity_scores.cpu().detach().numpy()
    top_diagnosis_idx = np.argsort(scores)[-top_n:][::-1]  # Get top N diagnoses
 
    # Return top N predictions for diagnosis
    return [medical_data[i] for i in top_diagnosis_idx]
 
# Get the top N most relevant diagnoses based on patient input
predicted_diagnoses = predict_medical_condition(patient_info, medical_data, top_n=2)
 
# Display the predicted diagnoses
print("Top Predicted Diagnoses:")
for diagnosis in predicted_diagnoses:
    print(f"Diagnosis: {diagnosis['text']}")