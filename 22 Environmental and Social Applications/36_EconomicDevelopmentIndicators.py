"""
Project 876: Economic Development Indicators
Description
Tracking economic development helps evaluate regional progress and direct investments effectively. In this project, we simulate macro- and micro-economic features and build a multi-output regression model to predict key development indicators: GDP per capita, unemployment rate, and human development index (HDI).

✅ Great for:

National development dashboards

Global competitiveness indexes

Aid and investment planning tools
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate economic and infrastructure data
np.random.seed(42)
n_samples = 1000
 
education_index = np.random.uniform(0.4, 1.0, n_samples)
internet_penetration = np.random.normal(0.6, 0.15, n_samples)     # 0–1 scale
industrial_index = np.random.normal(0.5, 0.2, n_samples)          # 0–1
infrastructure_score = np.random.uniform(0.3, 1.0, n_samples)
urbanization_rate = np.random.normal(0.65, 0.1, n_samples)        # 0–1
 
# Simulate outputs:
# GDP per capita (USD), Unemployment rate (%), HDI (0–1)
gdp_per_capita = 3000 + 10000 * education_index + 5000 * industrial_index + np.random.normal(0, 500, n_samples)
unemployment_rate = 0.3 - 0.1 * education_index - 0.05 * internet_penetration + np.random.normal(0, 0.02, n_samples)
unemployment_rate = np.clip(unemployment_rate, 0, 1)
hdi = 0.4 * education_index + 0.2 * internet_penetration + 0.2 * infrastructure_score + 0.2 * urbanization_rate + np.random.normal(0, 0.02, n_samples)
hdi = np.clip(hdi, 0, 1)
 
# Feature matrix and labels
X = np.stack([education_index, internet_penetration, industrial_index, infrastructure_score, urbanization_rate], axis=1)
y = np.stack([gdp_per_capita, unemployment_rate, hdi], axis=1)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-output regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3)  # Outputs: GDP per capita, Unemployment, HDI
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Development Indicator Prediction MAE: {mae}")
 
# Predict for 5 sample regions
preds = model.predict(X_test[:5])
for i in range(5):
    print(f"\nRegion {i+1} Predictions:")
    print(f"  💰 GDP per capita: ${preds[i][0]:,.0f}")
    print(f"  📉 Unemployment rate: {preds[i][1]*100:.1f}%")
    print(f"  📊 HDI: {preds[i][2]:.3f}")