"""
Project 818. Predictive Maintenance System

A predictive maintenance system anticipates equipment failures before they occur using historical sensor data. By predicting failures early, businesses can reduce downtime and maintenance costs. In this basic example, we simulate sensor readings and use logistic regression to classify whether a machine is likely to fail.

This model trains on past sensor data to predict whether a machine is at risk of failure. You can extend this with time-series models, anomaly detection, or survival analysis for more realistic predictive maintenance systems.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
# Simulated dataset: sensor readings and failure status (1 = failure, 0 = normal)
data = {
    'Temperature': [70, 75, 80, 85, 90, 95, 100, 105],
    'Vibration': [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
    'RPM': [1500, 1600, 1650, 1700, 1800, 1850, 1900, 1950],
    'Failed': [0, 0, 0, 0, 1, 1, 1, 1]
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df[['Temperature', 'Vibration', 'RPM']]
y = df['Failed']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Predict failure
y_pred = model.predict(X_test)
 
# Output model performance
print("Predictive Maintenance Classification Report:")
print(classification_report(y_test, y_pred))
 
# Example: Predict failure for new sensor reading
new_reading = pd.DataFrame({'Temperature': [92], 'Vibration': [0.9], 'RPM': [1820]})
failure_risk = model.predict_proba(new_reading)[0][1]
print(f"\nPredicted Failure Risk: {failure_risk:.2%}")