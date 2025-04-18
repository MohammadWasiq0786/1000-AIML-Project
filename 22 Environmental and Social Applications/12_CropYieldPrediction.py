"""
Project 852: Crop Yield Prediction
Description
Crop yield prediction helps farmers and policymakers make data-driven decisions about planting, irrigation, and resource allocation. In this project, we simulate agricultural features (soil quality, rainfall, temperature, etc.) and build a regression model to predict crop yield (tons/hectare).

✅ This model can:

Be integrated with drone or satellite data

Feed into agriculture dashboards

Guide precision farming practices
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate agricultural data
np.random.seed(42)
n_samples = 1000
 
soil_quality = np.random.normal(7, 1.5, n_samples)          # 1–10 scale
rainfall = np.random.normal(300, 50, n_samples)             # mm/month
temperature = np.random.normal(25, 3, n_samples)            # °C
fertilizer_use = np.random.normal(150, 30, n_samples)       # kg/hectare
pesticide_use = np.random.normal(1.2, 0.3, n_samples)       # liters/hectare
 
# Simulate crop yield (tons/hectare)
yield_output = (
    0.4 * soil_quality +
    0.02 * rainfall -
    0.1 * abs(temperature - 25) +
    0.05 * fertilizer_use -
    0.2 * pesticide_use +
    np.random.normal(0, 1, n_samples)  # noise
)
 
# Combine features
X = np.stack([soil_quality, rainfall, temperature, fertilizer_use, pesticide_use], axis=1)
y = yield_output
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Predicted yield
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Crop Yield Prediction MAE: {mae:.2f} tons/hectare")
 
# Predict for a few samples and plot
predictions = model.predict(X_test[:50]).flatten()
plt.figure(figsize=(10, 4))
plt.plot(y_test[:50], label="Actual Yield")
plt.plot(predictions, label="Predicted Yield")
plt.title("Crop Yield Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Yield (tons/hectare)")
plt.legend()
plt.grid(True)
plt.show()