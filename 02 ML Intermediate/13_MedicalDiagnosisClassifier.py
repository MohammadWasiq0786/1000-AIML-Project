"""
Project 53. Medical diagnosis classifier
Description:
A medical diagnosis classifier predicts the likelihood of a disease or condition based on patient data. In this project, we simulate a binary classification model that predicts the presence of a medical condition (e.g., diabetes) using features like age, glucose level, BMI, and blood pressure. We use a Logistic Regression model from scikit-learn.

About:
🩺 What This Project Demonstrates:
Builds a binary classifier for medical condition diagnosis

Uses standardized features and evaluates with confusion matrix, ROC-AUC, and classification report

Handles healthcare-specific data features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
 
# Simulated medical dataset
data = {
    'Age': [45, 50, 25, 30, 60, 35, 40, 55, 29, 48],
    'Glucose': [150, 180, 95, 105, 200, 110, 145, 190, 85, 175],
    'BMI': [28.0, 32.5, 22.0, 24.5, 35.0, 26.0, 29.5, 31.5, 23.0, 30.0],
    'BloodPressure': [80, 85, 70, 75, 90, 78, 82, 88, 72, 83],
    'Condition': [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]  # 1 = Positive diagnosis
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df.drop('Condition', axis=1)
y = df['Condition']
 
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
 
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix - Medical Diagnosis")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
 
# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.2f}")