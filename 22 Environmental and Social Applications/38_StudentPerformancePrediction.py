"""
Project 878: Student Performance Prediction
Description
Predicting student performance helps identify at-risk students and guide personalized interventions. In this project, we simulate academic and demographic data and build a regression model to predict student scores based on features such as study time, parental involvement, and school facilities.

✅ This model can be used for:

Personalized student intervention systems

Early identification of at-risk students

School performance evaluation dashboards
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate student academic and demographic data
np.random.seed(42)
n_samples = 1000
 
study_time = np.random.normal(4, 2, n_samples)                 # hours/week
parental_involvement = np.random.uniform(0, 1, n_samples)     # 0–1 scale
school_quality_score = np.random.uniform(0, 1, n_samples)     # 0–1 scale
sleep_hours = np.random.normal(7, 1.5, n_samples)             # hours/day
previous_grades = np.random.normal(75, 10, n_samples)         # percentage
 
# Simulate student performance (final exam score)
performance_score = (
    0.4 * previous_grades +
    0.3 * study_time * 10 +  # study time * weight
    0.2 * parental_involvement * 20 +
    0.1 * school_quality_score * 30 +
    np.random.normal(0, 5, n_samples)  # noise
)
 
# Feature matrix and labels
X = np.stack([study_time, parental_involvement, school_quality_score, sleep_hours, previous_grades], axis=1)
y = performance_score
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model for student performance prediction
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output: predicted score
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Student Performance Prediction MAE: {mae:.2f} points")
 
# Predict for 5 students
preds = model.predict(X_test[:5]).flatten()
for i in range(5):
    print(f"\nStudent {i+1}: Predicted Score = {preds[i]:.1f} (Actual: {y_test[i]:.1f})")