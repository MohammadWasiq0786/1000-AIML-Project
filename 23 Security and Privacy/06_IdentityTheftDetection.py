"""
Project 886. Identity Theft Detection

Identity theft detection focuses on spotting unusual behavior that suggests someone is using a person's credentials fraudulently. This project uses user behavior data (e.g., location, device, transaction type) and builds a binary classifier to detect identity theft attempts.

This model flags suspicious access patterns (e.g., odd login times, new locations, or unrecognized devices) that may suggest identity theft. For enhanced detection:

Use sequence analysis (login timelines)

Integrate IP, GPS, and biometric data

Monitor account behavior drift using clustering or anomaly detection
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated identity-related transaction logs
data = {
    'LoginHour': [10, 2, 14, 1, 13, 0, 15, 3],             # time of login
    'DeviceMatch': [1, 0, 1, 0, 1, 0, 1, 0],               # known device (1) or not (0)
    'LocationMatch': [1, 0, 1, 0, 1, 0, 1, 0],             # known location (1) or not (0)
    'MultipleAttempts': [0, 1, 0, 1, 0, 1, 0, 1],          # repeated login attempts
    'IdentityTheft': [0, 1, 0, 1, 0, 1, 0, 1]              # 1 = identity theft, 0 = normal
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df.drop('IdentityTheft', axis=1)
y = df['IdentityTheft']
 
# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train the identity theft detection model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
print("Identity Theft Detection Report:")
print(classification_report(y_test, y_pred))
 
# Predict on new login behavior
new_login = pd.DataFrame([{
    'LoginHour': 1,
    'DeviceMatch': 0,
    'LocationMatch': 0,
    'MultipleAttempts': 1
}])
 
risk_score = model.predict_proba(new_login)[0][1]
print(f"\nPredicted Identity Theft Risk: {risk_score:.2%}")