"""
Project 861: Pollution Source Identification
Description
Identifying the source of pollution is critical for effective environmental management. This project simulates environmental sensor data (e.g., air, water, or soil readings) from multiple zones and builds a multi-class classification model to predict the type of pollution source (e.g., Industrial, Agricultural, Urban Runoff, Natura

✅ This model supports:

Environmental audit systems

Government pollution tracking

Smart sensor grids across rivers, lakes, or air quality zones
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate environmental readings
np.random.seed(42)
n_samples = 1000
 
# Features: nitrate level, phosphate level, heavy metals, turbidity, BOD (biochemical oxygen demand)
nitrate = np.random.normal(5, 2, n_samples)            # mg/L
phosphate = np.random.normal(0.5, 0.3, n_samples)      # mg/L
heavy_metals = np.random.normal(0.05, 0.03, n_samples) # mg/L
turbidity = np.random.normal(20, 10, n_samples)        # NTU
bod = np.random.normal(3, 1.5, n_samples)              # mg/L
 
# Labels: 0 = Industrial, 1 = Agricultural, 2 = Urban Runoff, 3 = Natural
source_type = np.where(
    (heavy_metals > 0.08), 0,  # Industrial
    np.where((nitrate > 7) & (phosphate > 0.7), 1,  # Agricultural
    np.where((turbidity > 25) & (bod > 4), 2, 3))   # Urban or Natural
)
 
# Combine features
X = np.stack([nitrate, phosphate, heavy_metals, turbidity, bod], axis=1)
y = source_type
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-class classification model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 pollution sources
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate the model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Pollution Source Identification Accuracy: {acc:.4f}")
 
# Predict for 5 environmental samples
preds = np.argmax(model.predict(X_test[:5]), axis=1)
source_map = {
    0: "🏭 Industrial",
    1: "🌾 Agricultural",
    2: "🏙️ Urban Runoff",
    3: "🌿 Natural"
}
 
for i in range(5):
    print(f"Sample {i+1}: Predicted = {source_map[preds[i]]}, Actual = {source_map[y_test[i]]}")