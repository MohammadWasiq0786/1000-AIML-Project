"""
Project 829. Interview Performance Prediction

Interview performance prediction helps estimate a candidate’s success based on pre-interview data (e.g., resume features, assessments, and behavioral traits). In this simplified project, we simulate candidate attributes and use a classification model to predict interview outcome (pass/fail).

This model uses candidate features to predict the likelihood of passing an interview. In real scenarios, additional data like test scores, behavioral assessments, or video interview analysis can be incorporated.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
 
# Simulated candidate dataset
data = {
    'TechnicalScore': [85, 60, 78, 92, 55, 65, 70, 88],
    'CommunicationScore': [80, 65, 75, 90, 50, 60, 68, 85],
    'ExperienceYears': [3, 1, 2, 4, 0, 1, 2, 5],
    'ProblemSolving': [4, 3, 4, 5, 2, 3, 3, 5],  # scale 1–5
    'PassedInterview': [1, 0, 1, 1, 0, 0, 0, 1]  # 1 = passed, 0 = failed
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df.drop('PassedInterview', axis=1)
y = df['PassedInterview']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Evaluate model
y_pred = model.predict(X_test)
print("Interview Performance Prediction Report:")
print(classification_report(y_test, y_pred))
 
# Predict new candidate's outcome
new_candidate = pd.DataFrame([{
    'TechnicalScore': 82,
    'CommunicationScore': 78,
    'ExperienceYears': 2,
    'ProblemSolving': 4
}])
 
prob_pass = model.predict_proba(new_candidate)[0][1]
print(f"\nPredicted Probability of Passing: {prob_pass:.2%}")