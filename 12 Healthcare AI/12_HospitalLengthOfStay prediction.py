"""
Project 452. Hospital length of stay prediction
Description:
Predicting length of stay (LOS) helps hospitals optimize bed management, staff scheduling, and resource allocation. In this project, we’ll create a regression model using structured clinical features to predict the number of days a patient will remain hospitalized.

About:
✅ What It Does:
Simulates a structured clinical dataset with LOS as a target.

Uses Gradient Boosting Regression for accurate predictions.

Can be extended to:

Predict ICU stay duration vs ward stay

Build interactive prediction dashboards

Add temporal features (e.g., day of admission, seasonal trends)

You can later use real datasets like:

MIMIC-III/MIMIC-IV (via PhysioNet)

Premier Healthcare Database
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
 
# 1. Simulated hospital admission data
np.random.seed(42)
data = {
    'age': np.random.randint(18, 90, 500),
    'gender': np.random.choice(['Male', 'Female'], 500),
    'diagnosis': np.random.choice(['Heart Failure', 'Pneumonia', 'Infection'], 500),
    'num_medications': np.random.randint(1, 40, 500),
    'num_lab_procedures': np.random.randint(5, 100, 500),
    'severity_score': np.random.randint(1, 5, 500),  # 1 (low) to 4 (high)
    'length_of_stay': np.random.normal(loc=7, scale=3, size=500).astype(int)  # target in days
}
df = pd.DataFrame(data)
df['length_of_stay'] = df['length_of_stay'].clip(lower=1)  # minimum 1 day
 
# 2. Encode categorical features
df['gender'] = LabelEncoder().fit_transform(df['gender'])
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])
 
# 3. Prepare data
X = df.drop('length_of_stay', axis=1)
y = df['length_of_stay']
 
# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 6. Train regression model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# 7. Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))