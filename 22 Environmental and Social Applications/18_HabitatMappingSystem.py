"""
Project 858: Habitat Mapping System
Description
A habitat mapping system uses environmental and geospatial data to classify land areas into habitat types (e.g., forest, grassland, wetland). It helps with land use planning, conservation, and biodiversity tracking. In this project, we simulate remote sensing features and build a multi-class classifier to predict habitat type.

✅ This model supports:

Land use classification

Conservation area zoning

Integration with drone/satellite imagery and GIS systems
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate remote sensing features
np.random.seed(42)
n_samples = 1000
 
ndvi = np.random.normal(0.6, 0.2, n_samples)              # Vegetation index
elevation = np.random.normal(300, 100, n_samples)         # meters
soil_moisture = np.random.uniform(0, 1, n_samples)        # 0–1 scale
proximity_to_water = np.random.normal(1.5, 0.8, n_samples)  # km
surface_temp = np.random.normal(28, 3, n_samples)         # °C
 
# Habitat type: 0 = Forest, 1 = Grassland, 2 = Wetland
habitat_type = np.where(
    (ndvi > 0.7) & (soil_moisture > 0.5), 0,      # Forest
    np.where((ndvi < 0.4) & (elevation > 350), 1, # Grassland
    2)                                           # Wetland (default)
)
 
# Feature matrix
X = np.stack([ndvi, elevation, soil_moisture, proximity_to_water, surface_temp], axis=1)
y = habitat_type
 
# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-class classifier
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 habitat classes
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Habitat Mapping Model Accuracy: {acc:.4f}")
 
# Predict habitat type for 5 locations
preds = np.argmax(model.predict(X_test[:5]), axis=1)
habitat_map = {0: "🌲 Forest", 1: "🌾 Grassland", 2: "🌊 Wetland"}
 
for i in range(5):
    print(f"Location {i+1}: Predicted = {habitat_map[preds[i]]}, Actual = {habitat_map[y_test[i]]}")