"""
Project 905. Fake Account Detection

Fake account detection helps platforms identify accounts created for spam, fraud, or manipulation. In this project, we simulate user profile data and use a classification model to flag accounts as fake or genuine based on features like profile completeness, activity level, and creation timing.

Key Features:
Profile completeness (picture, bio)

Account age

Activity level

🧠 In real systems:

Use IP/email/device fingerprints

Combine behavioral features (e.g., burst posting, friend patterns)

Leverage graph analysis for coordinated networks
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated user profile dataset
data = {
    'ProfilePicture': [1, 0, 1, 0, 0, 1, 0, 1],     # has profile pic (1/0)
    'BioLength': [80, 5, 100, 0, 10, 120, 3, 90],   # bio character length
    'DaysSinceCreation': [365, 2, 400, 1, 3, 500, 0, 300],
    'PostsMade': [150, 1, 200, 0, 2, 250, 0, 180],
    'FakeAccount': [0, 1, 0, 1, 1, 0, 1, 0]         # 1 = fake, 0 = real
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df.drop('FakeAccount', axis=1)
y = df['FakeAccount']
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
print("Fake Account Detection Report:")
print(classification_report(y_test, y_pred))
 
# Predict on new account
new_account = pd.DataFrame([{
    'ProfilePicture': 0,
    'BioLength': 7,
    'DaysSinceCreation': 1,
    'PostsMade': 0
}])
 
risk_score = model.predict_proba(new_account)[0][1]
print(f"\nPredicted Fake Account Risk: {risk_score:.2%}")