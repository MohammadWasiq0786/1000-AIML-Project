"""
Project 830. Employee Performance Prediction

Employee performance prediction uses HR and work-related data to forecast how well an employee might perform. This can support decisions in hiring, promotion, and training. In this example, we’ll build a classification model to predict high or low performance based on work experience, training hours, and prior ratings.

This predictive model can help HR teams assess candidates or current employees, and personalize development plans. It can be enhanced with peer feedback, KPI trends, or even sentiment analysis from internal communications.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated dataset of employee attributes and performance outcome
data = {
    'ExperienceYears': [1, 5, 2, 4, 3, 6, 2, 7],
    'TrainingHours': [20, 10, 30, 15, 25, 8, 22, 5],
    'PreviousRating': [3, 5, 4, 4, 3, 5, 3, 5],  # 1–5 scale
    'Certifications': [0, 1, 1, 1, 0, 1, 0, 1],
    'Performance': [0, 1, 1, 1, 0, 1, 0, 1]  # 1 = high performer, 0 = low performer
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df.drop('Performance', axis=1)
y = df['Performance']
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Evaluate model
y_pred = model.predict(X_test)
print("Employee Performance Prediction Report:")
print(classification_report(y_test, y_pred))
 
# Predict performance for a new employee
new_employee = pd.DataFrame([{
    'ExperienceYears': 3,
    'TrainingHours': 18,
    'PreviousRating': 4,
    'Certifications': 1
}])
 
prob_high_perf = model.predict_proba(new_employee)[0][1]
print(f"\nPredicted Probability of High Performance: {prob_high_perf:.2%}")