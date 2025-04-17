"""
Project 821. Fraud Detection for Businesses

Fraud detection systems help identify unusual transactions or behaviors that may indicate fraud. In this basic project, we'll simulate transaction data and train a classification model to detect fraudulent activity using features like transaction amount, frequency, and location deviation.

This model learns from patterns in transaction data to classify future transactions as fraudulent or not. It can be extended with time-series patterns, user profiling, and deep learning for higher complexity fraud detection pipelines.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated transaction dataset
data = {
    'Amount': [20, 500, 15, 2000, 25, 4500, 35, 3000, 45, 10],
    'Frequency': [5, 1, 6, 1, 7, 1, 5, 1, 4, 8],  # transactions per day
    'IsForeignTransaction': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    'IsHighRiskCountry': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    'IsWeekend': [0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    'Fraud': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]  # 1 = fraud, 0 = normal
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df.drop('Fraud', axis=1)
y = df['Fraud']
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Predict on test set
y_pred = model.predict(X_test)
 
# Evaluate performance
print("Fraud Detection Model Evaluation:")
print(classification_report(y_test, y_pred))
 
# Predict on new transaction
new_transaction = pd.DataFrame([{
    'Amount': 3500,
    'Frequency': 1,
    'IsForeignTransaction': 1,
    'IsHighRiskCountry': 1,
    'IsWeekend': 0
}])
fraud_prob = model.predict_proba(new_transaction)[0][1]
print(f"\nPredicted Fraud Probability: {fraud_prob:.2%}")