"""
Project 831. Employee Attrition Prediction

Employee attrition prediction identifies individuals likely to leave an organization, enabling proactive retention strategies. In this project, we simulate HR data and use a classification model to predict attrition based on factors like satisfaction, workload, and years at the company.

This model estimates the risk of an employee leaving based on behavioral and organizational metrics. In real-world scenarios, it can be enriched with survey responses, salary changes, or manager feedback.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated HR dataset
data = {
    'SatisfactionLevel': [0.9, 0.4, 0.8, 0.3, 0.6, 0.7, 0.2, 0.85],
    'AverageMonthlyHours': [180, 240, 160, 250, 200, 170, 260, 155],
    'YearsAtCompany': [3, 2, 4, 1, 5, 3, 1, 6],
    'PromotionLast5Years': [0, 0, 1, 0, 1, 0, 0, 1],
    'Attrition': [0, 1, 0, 1, 0, 0, 1, 0]  # 1 = left, 0 = stayed
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Predict attrition
y_pred = model.predict(X_test)
 
# Evaluate model
print("Employee Attrition Prediction Report:")
print(classification_report(y_test, y_pred))
 
# Predict for a new employee
new_employee = pd.DataFrame([{
    'SatisfactionLevel': 0.5,
    'AverageMonthlyHours': 230,
    'YearsAtCompany': 2,
    'PromotionLast5Years': 0
}])
 
attrition_risk = model.predict_proba(new_employee)[0][1]
print(f"\nPredicted Attrition Risk: {attrition_risk:.2%}")