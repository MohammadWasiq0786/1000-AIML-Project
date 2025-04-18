"""
Project 854: Sustainable Farming Recommendation
Description
A Sustainable Farming Recommendation System suggests eco-friendly practices based on land, crop type, climate, and resource usage. In this project, we simulate agricultural and environmental inputs and build a multi-class classification model to recommend a sustainable farming strategy, such as organic practices, crop rotation, drip irrigation, etc.

✅ Real-world integrations:

Farmer advisory apps

Government agri-portals

AI-powered smart farming systems
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
 
# Simulate input features: soil type index (0–4), water availability (%), average temp (°C), crop type index (0–4)
np.random.seed(42)
n_samples = 1000
 
soil_type = np.random.randint(0, 5, n_samples)
water_availability = np.random.normal(60, 20, n_samples)      # %
avg_temperature = np.random.normal(26, 3, n_samples)          # °C
crop_type = np.random.randint(0, 5, n_samples)
 
# Simulate 4 recommendations based on heuristic logic
# 0 = crop rotation, 1 = organic farming, 2 = drip irrigation, 3 = compost use
recommendation = np.where(
    (soil_type == 2) & (water_availability < 50), 2,
    np.where((crop_type == 1) & (avg_temperature > 28), 1,
    np.where((soil_type == 4) & (water_availability > 70), 0, 3))
)
 
# Stack features
X = np.stack([soil_type, water_availability, avg_temperature, crop_type], axis=1)
y = recommendation
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build model for multi-class classification
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 classes
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Sustainable Farming Recommender Accuracy: {acc:.4f}")
 
# Predict for a few samples
recommend_map = {
    0: "🌾 Crop Rotation",
    1: "🍀 Organic Farming",
    2: "💧 Drip Irrigation",
    3: "🌱 Compost Usage"
}
preds = np.argmax(model.predict(X_test[:5]), axis=1)
for i in range(5):
    print(f"Sample {i+1}: Recommended = {recommend_map[preds[i]]} (Actual: {recommend_map[y_test[i]]})")