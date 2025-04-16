"""
Project 465. Sleep quality analysis
Description:
Sleep quality analysis uses data from sleep trackers, motion sensors, or wearables to assess sleep stages (light/deep/REM), interruptions, and overall restfulness. In this project, we'll simulate a simple analysis system that takes nightly logs and predicts sleep quality score using a regression model.

About:
✅ What It Does:
Simulates sleep metrics and predicts a sleep quality score.

Uses linear regression, easily replaceable with Random Forest, XGBoost, or neural networks.

Can be extended to:

Use real wearable sensor data

Visualize sleep trends over time

Trigger alerts or suggestions based on poor sleep history

For real-world systems:

Integrate with Fitbit, Oura, Apple Health, or Polysomnography data

Use machine learning on data like HRV, movement, snoring, and sleep stages
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
 
# 1. Simulated sleep dataset
np.random.seed(42)
data = {
    "total_sleep_time": np.random.normal(6.5, 1.0, 100),         # in hours
    "deep_sleep_time": np.random.normal(1.2, 0.4, 100),          # in hours
    "number_of_awakenings": np.random.randint(0, 6, 100),        
    "snoring_intensity": np.random.uniform(0, 1, 100),           
    "sleep_efficiency": np.random.normal(85, 10, 100),           # %
    "sleep_quality_score": np.random.normal(75, 10, 100)         # target score out of 100
}
df = pd.DataFrame(data)
 
# 2. Features and target
X = df.drop("sleep_quality_score", axis=1)
y = df["sleep_quality_score"]
 
# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 4. Train regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# 5. Evaluate model
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Mean Absolute Error in Sleep Score Prediction: {mae:.2f}")
 
# 6. Predict on new input
new_sample = pd.DataFrame([{
    "total_sleep_time": 7.0,
    "deep_sleep_time": 1.4,
    "number_of_awakenings": 2,
    "snoring_intensity": 0.3,
    "sleep_efficiency": 90
}])
predicted_score = model.predict(new_sample)[0]
print(f"\nPredicted Sleep Quality Score: {predicted_score:.1f} / 100")