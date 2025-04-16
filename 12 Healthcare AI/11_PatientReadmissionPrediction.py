"""
Project 451. Patient readmission prediction
Description:
Patient readmission prediction helps hospitals reduce costs and improve care quality by identifying patients at high risk of being readmitted within 30 days of discharge. This project uses structured clinical data (e.g., diagnosis codes, length of stay, demographics) to build a binary classifier predicting readmission likelihood.

About:
✅ What It Does:
Simulates a structured hospital dataset.

Uses a Random Forest to predict patient readmission risk.

Easily extendable to:

Include lab values, comorbidities, insurance status

Build a real-time dashboard for care providers

Add explainability (e.g., SHAP values)

You can later use real datasets like:

MIMIC-III or MIMIC-IV (PhysioNet ICU records)

Hospital Readmissions Reduction Program (HRRP) data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
 
# 1. Simulated dataset (structured clinical data)
np.random.seed(42)
data = {
    'age': np.random.randint(20, 90, size=500),
    'length_of_stay': np.random.randint(1, 15, size=500),
    'num_lab_procedures': np.random.randint(5, 80, size=500),
    'num_medications': np.random.randint(1, 50, size=500),
    'diag_primary': np.random.choice(['Diabetes', 'Heart Failure', 'Pneumonia'], size=500),
    'gender': np.random.choice(['Male', 'Female'], size=500),
    'readmitted': np.random.choice([0, 1], size=500)  # 1 = readmitted within 30 days
}
df = pd.DataFrame(data)
 
# 2. Encode categorical features
df['gender'] = LabelEncoder().fit_transform(df['gender'])
df['diag_primary'] = LabelEncoder().fit_transform(df['diag_primary'])
 
# 3. Split features/labels
X = df.drop('readmitted', axis=1)
y = df['readmitted']
 
# 4. Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 6. Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# 7. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))