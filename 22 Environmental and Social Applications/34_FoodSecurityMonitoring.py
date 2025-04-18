"""
Project 874: Food Security Monitoring
Description
Food security monitoring helps identify regions at risk of hunger or malnutrition due to environmental, economic, or supply factors. In this project, we simulate socioeconomic and environmental indicators and build a binary classification model to flag regions as food secure or insecure.

✅ Use Cases:

UN food security dashboards

Early warning systems

Agricultural policy planning
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate regional data
np.random.seed(42)
n_samples = 1000
 
crop_yield = np.random.normal(2.5, 0.8, n_samples)               # tons/hectare
rainfall = np.random.normal(100, 30, n_samples)                  # mm/month
market_access_score = np.random.uniform(0, 1, n_samples)         # 0–1
poverty_rate = np.random.normal(0.3, 0.1, n_samples)             # 0–1
food_price_index = np.random.normal(120, 20, n_samples)          # relative price index
 
# Label: 1 = food insecure if low yield, high poverty, poor access
food_insecure = (
    (crop_yield < 2.0) &
    (poverty_rate > 0.35) &
    (market_access_score < 0.4)
).astype(int)
 
# Combine features
X = np.stack([
    crop_yield,
    rainfall,
    market_access_score,
    poverty_rate,
    food_price_index
], axis=1)
y = food_insecure
 
# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build binary classification model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 1 = food insecure
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Food Security Classifier Accuracy: {acc:.4f}")
 
# Predict for 5 regions
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Region {i+1}: {'⚠️ Food Insecure' if preds[i] else '✅ Food Secure'} (Actual: {'⚠️' if y_test[i] else '✅'})")