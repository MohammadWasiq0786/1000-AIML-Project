"""
Project 819. Equipment Failure Prediction

Equipment failure prediction aims to forecast when a machine or system might fail based on operational and environmental data. Unlike binary classification in maintenance alerts, this example focuses on regression to predict time-to-failure (TTF) — useful for planning proactive maintenance windows.

This model estimates how much time is left before equipment fails based on real-time sensor inputs. It's valuable for maintenance scheduling, especially in industrial and critical infrastructure settings.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
 
# Simulated dataset: sensor readings and remaining time before failure in hours
data = {
    'Temperature': [70, 75, 80, 85, 90, 95, 100, 105],
    'Pressure': [30, 32, 35, 38, 42, 45, 48, 50],
    'Vibration': [0.2, 0.3, 0.4, 0.6, 0.8, 1.1, 1.3, 1.6],
    'TimeToFailure': [48, 40, 35, 28, 20, 14, 7, 2]  # target: hours left before failure
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df[['Temperature', 'Pressure', 'Vibration']]
y = df['TimeToFailure']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predict time to failure
y_pred = model.predict(X_test)
 
# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} hours")
 
# Example prediction for new input
new_input = pd.DataFrame({'Temperature': [92], 'Pressure': [43], 'Vibration': [0.85]})
predicted_ttf = model.predict(new_input)[0]
print(f"Predicted Time to Failure: {predicted_ttf:.2f} hours")