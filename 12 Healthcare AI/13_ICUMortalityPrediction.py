"""
Project 453. ICU mortality prediction
Description:
ICU Mortality Prediction is used to assess a critically ill patient's likelihood of survival based on clinical features like vitals, lab results, comorbidities, and SOFA scores. This project builds a binary classification model to predict whether an ICU patient will survive or not, using structured data.

About:
✅ What It Does:
Simulates ICU patient data including vital signs and scores.

Trains a Random Forest to predict mortality (0 = survived, 1 = deceased).

Can be extended to:

Add time series vitals (LSTM or transformer model)

Use real ICU datasets like MIMIC or eICU

Implement model explainability (e.g., SHAP) for clinical decisions


You can later use real-world ICU data from:

MIMIC-III / MIMIC-IV on PhysioNet

eICU Collaborative Research Database
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
 
# 1. Simulated ICU patient data
np.random.seed(42)
data = {
    'age': np.random.randint(18, 90, 500),
    'gender': np.random.choice(['Male', 'Female'], 500),
    'sofa_score': np.random.randint(0, 20, 500),
    'glucose': np.random.normal(110, 20, 500),
    'heart_rate': np.random.normal(85, 10, 500),
    'systolic_bp': np.random.normal(120, 15, 500),
    'spo2': np.random.normal(96, 3, 500),
    'mortality': np.random.choice([0, 1], 500, p=[0.75, 0.25])  # 0 = survived, 1 = died
}
df = pd.DataFrame(data)
 
# 2. Encode categorical
df['gender'] = LabelEncoder().fit_transform(df['gender'])
 
# 3. Features and label
X = df.drop('mortality', axis=1)
y = df['mortality']
 
# 4. Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 6. Train binary classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# 7. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Survived', 'Deceased']))