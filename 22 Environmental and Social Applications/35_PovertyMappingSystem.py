"""
Project 875: Poverty Mapping System
Description
A poverty mapping system helps identify underdeveloped or at-risk regions based on socioeconomic indicators. In this project, we simulate regional data and build a regression model to estimate the poverty rate of each area, which can be visualized in geospatial dashboards for resource allocation and policy design.

✅ Applications include:

Government welfare planning

Nonprofit development targeting

Global inequality dashboards
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate regional development data
np.random.seed(42)
n_samples = 1000
 
education_index = np.random.uniform(0, 1, n_samples)            # 0–1 (higher is better)
employment_rate = np.random.normal(0.65, 0.1, n_samples)        # %
access_to_services = np.random.uniform(0, 1, n_samples)         # 0–1
infrastructure_score = np.random.uniform(0, 1, n_samples)       # 0–1
healthcare_access = np.random.uniform(0, 1, n_samples)          # 0–1
 
# Target variable: poverty rate (0 to 1)
poverty_rate = (
    1 - education_index * 0.3 -
    employment_rate * 0.4 -
    access_to_services * 0.1 -
    infrastructure_score * 0.1 -
    healthcare_access * 0.1 +
    np.random.normal(0, 0.05, n_samples)
)
poverty_rate = np.clip(poverty_rate, 0, 1)
 
# Combine features
X = np.stack([
    education_index,
    employment_rate,
    access_to_services,
    infrastructure_score,
    healthcare_access
], axis=1)
y = poverty_rate
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output: predicted poverty rate (0–1)
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate performance
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Poverty Rate Prediction MAE: {mae:.4f}")
 
# Predict for 5 sample regions
preds = model.predict(X_test[:5]).flatten()
for i in range(5):
    print(f"Region {i+1}: Predicted Poverty Rate = {preds[i]:.2f} (Actual: {y_test[i]:.2f})")