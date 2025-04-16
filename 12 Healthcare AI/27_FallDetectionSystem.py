"""
Project 467. Fall detection system
Description:
A Fall Detection System automatically detects if a person has fallen, using data from accelerometers, gyroscopes, or camera feeds. This project simulates a system that uses wearable sensor data (x, y, z acceleration) and classifies events as fall or normal activity using a machine learning model.

About:
✅ What It Does:
Simulates 3-axis accelerometer data.

Calculates motion magnitude to detect sudden falls.

Uses a Random Forest classifier for accurate fall vs. normal detection.

Extendable to:

Real-time monitoring on Raspberry Pi, Edge AI

Include gyroscope/angle changes

Trigger SMS/emergency alerts on fall detection

For real-world applications:

Use datasets like SisFall, MobiAct, or UR Fall Detection

Integrate with smartwatches, IoT devices, or CCTV feeds
"""


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# 1. Simulated sensor dataset: [x, y, z, magnitude], label (1=fall, 0=normal)
np.random.seed(42)
samples = 500
data = {
    "x": np.random.normal(0, 1, samples),
    "y": np.random.normal(0, 1, samples),
    "z": np.random.normal(9.8, 1, samples),
}
df = pd.DataFrame(data)
df["magnitude"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
df["label"] = (df["magnitude"] > 15).astype(int)  # fall if sudden spike in acceleration
 
# 2. Features and labels
X = df[["x", "y", "z", "magnitude"]]
y = df["label"]
 
# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 4. Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# 5. Evaluate
y_pred = model.predict(X_test)
print("Fall Detection Report:\n")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fall"]))
 
# 6. Predict on new event
new_event = pd.DataFrame([{"x": 2.3, "y": 1.7, "z": 19.5}])
new_event["magnitude"] = np.sqrt(new_event["x"]**2 + new_event["y"]**2 + new_event["z"]**2)
prediction = model.predict(new_event)[0]
print(f"\nPrediction for new event: {'FALL' if prediction == 1 else 'NORMAL'}")