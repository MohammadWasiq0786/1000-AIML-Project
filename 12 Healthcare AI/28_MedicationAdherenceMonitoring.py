"""
Project 468. Medication adherence monitoring
Description:
Medication adherence monitoring ensures patients are taking their medications on time and in the correct dosage. AI can use reminders, logs, or even computer vision to track adherence. In this project, we simulate a daily intake logging system and train a classifier to predict non-adherence risk based on patterns.

About:
✅ What It Does:
Tracks adherence behavior using binary indicators (taken, timing, side effects).

Uses a logistic regression model to predict non-adherence risk.

Can be extended to:

Include longitudinal adherence patterns

Send reminders or doctor alerts

Integrate with smart pillbox sensors

For real-world data:

Integrate with pill tracker apps, smart pill bottles, or EHRs

Use behavioral data (missed doses, timing delays, side effects)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# 1. Simulate medication adherence logs
np.random.seed(42)
data = {
    "dose_taken": np.random.choice([1, 0], 300, p=[0.85, 0.15]),
    "on_time": np.random.choice([1, 0], 300, p=[0.80, 0.20]),
    "missed_previous_day": np.random.choice([1, 0], 300, p=[0.10, 0.90]),
    "reported_side_effects": np.random.choice([1, 0], 300, p=[0.15, 0.85]),
    "adherence_risk": None
}
 
df = pd.DataFrame(data)
 
# 2. Assign target label: 1 = likely non-adherent
df["adherence_risk"] = (
    (df["dose_taken"] == 0) |
    (df["on_time"] == 0) |
    (df["missed_previous_day"] == 1) |
    (df["reported_side_effects"] == 1)
).astype(int)
 
# 3. Train/test split
X = df.drop("adherence_risk", axis=1)
y = df["adherence_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 4. Train model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# 5. Evaluate
y_pred = model.predict(X_test)
print("Medication Adherence Risk Report:\n")
print(classification_report(y_test, y_pred, target_names=["Adherent", "Non-Adherent"]))
 
# 6. Predict new user entry
new_user = pd.DataFrame([{
    "dose_taken": 1,
    "on_time": 0,
    "missed_previous_day": 0,
    "reported_side_effects": 1
}])
prediction = model.predict(new_user)[0]
print(f"\nPrediction: {'Non-Adherent' if prediction == 1 else 'Adherent'}")