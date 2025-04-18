"""
Project 884. Fraud Detection for Transactions

Fraud detection for transactions identifies suspicious financial activities in real time (e.g., unauthorized purchases, account takeovers). In this project, we simulate transaction data and use a binary classification model to detect fraud based on transaction patterns.

This model uses contextual features (amount, time, device, and location) to flag transactions as potentially fraudulent. It can be enhanced with:

Time series behavior modeling

Graph-based fraud networks

Deep learning for complex feature interactions
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated transaction dataset
data = {
    'Amount': [100, 5000, 200, 7000, 80, 15000, 50, 3000],  # transaction amount
    'TimeOfDay': [10, 23, 14, 1, 9, 2, 13, 22],             # hour of transaction
    'LocationMatch': [1, 0, 1, 0, 1, 0, 1, 0],              # does it match known location
    'DeviceTrusted': [1, 0, 1, 0, 1, 0, 1, 0],              # familiar device used
    'IsFraud': [0, 1, 0, 1, 0, 1, 0, 1]                     # 1 = fraud, 0 = genuine
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df.drop('IsFraud', axis=1)
y = df['IsFraud']
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train the fraud classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Evaluate model
y_pred = model.predict(X_test)
print("Transaction Fraud Detection Report:")
print(classification_report(y_test, y_pred))
 
# Predict on a new transaction
new_transaction = pd.DataFrame([{
    'Amount': 6800,
    'TimeOfDay': 1,
    'LocationMatch': 0,
    'DeviceTrusted': 0
}])
 
fraud_risk = model.predict_proba(new_transaction)[0][1]
print(f"\nPredicted Fraud Risk: {fraud_risk:.2%}")