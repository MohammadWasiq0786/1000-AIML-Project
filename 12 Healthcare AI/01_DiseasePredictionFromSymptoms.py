"""
Project 441. Disease prediction from symptoms
Description:
This project builds a machine learning model that takes a list of reported symptoms as input and predicts the most probable disease. It can serve as a basic triage tool or be integrated into a chatbot or health assistant.

About:
✅ What It Does:
Encodes symptom combinations into a multi-hot vector.

Trains a logistic regression model to predict the disease.

Includes a function for real-time prediction from symptoms.

Can be extended with real datasets like SymCAT, HealthMap, or scraped clinical notes.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# 1. Simulate dataset: symptoms → disease
data = [
    (["fever", "cough", "sore throat"], "Flu"),
    (["headache", "nausea", "dizziness"], "Migraine"),
    (["chest pain", "shortness of breath", "sweating"], "Heart Attack"),
    (["abdominal pain", "diarrhea", "vomiting"], "Food Poisoning"),
    (["fatigue", "weight loss", "frequent urination"], "Diabetes"),
    (["itching", "rash", "swelling"], "Allergy"),
    (["joint pain", "stiffness", "swelling"], "Arthritis"),
    (["back pain", "numbness", "tingling"], "Spinal Disc Problem"),
]
 
# Convert to DataFrame
df = pd.DataFrame(data, columns=["Symptoms", "Disease"])
 
# 2. Encode symptoms (multi-hot encoding)
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])
 
# 3. Encode labels
le = LabelEncoder()
y = le.fit_transform(df["Disease"])
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# 5. Train classifier
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
 
# 6. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
 
# 7. Inference: predict disease from user symptoms
def predict_disease(symptom_list):
    input_vec = mlb.transform([symptom_list])
    prediction = model.predict(input_vec)
    return le.inverse_transform(prediction)[0]
 
# Example
example_symptoms = ["fever", "sore throat"]
predicted_disease = predict_disease(example_symptoms)
print(f"Predicted Disease: {predicted_disease}")