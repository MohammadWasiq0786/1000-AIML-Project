"""
Project 885. Credit Card Fraud Detection

Credit card fraud detection systems identify unauthorized or suspicious card usage based on transaction patterns. In this project, we simulate anonymized credit card transaction data and build a binary classification model using LogisticRegression.

This model flags potentially fraudulent card activity based on timing, context, and mode of payment. For real-world deployment, you'd work with:

Highly imbalanced datasets (e.g., <1% fraud)

Feature engineering from transaction sequences

SMOTE, under-sampling, or anomaly detection techniques

Encrypted customer behavior signals
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated credit card transaction features
data = {
    'TransactionAmount': [25.5, 4500.0, 12.0, 8800.0, 33.0, 9100.0, 29.0, 4999.0],
    'TransactionHour': [14, 1, 12, 0, 13, 3, 11, 2],          # hour of the day (0-23)
    'IsOnline': [1, 1, 0, 1, 0, 1, 0, 1],                    # whether it was online
    'CardPresent': [1, 0, 1, 0, 1, 0, 1, 0],                 # physical card used or not
    'Fraud': [0, 1, 0, 1, 0, 1, 0, 1]                        # 1 = fraud, 0 = legitimate
}
 
df = pd.DataFrame(data)
 
# Features and label
X = df.drop('Fraud', axis=1)
y = df['Fraud']
 
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Evaluate model
y_pred = model.predict(X_test)
print("Credit Card Fraud Detection Report:")
print(classification_report(y_test, y_pred))
 
# Predict on a new credit card transaction
new_transaction = pd.DataFrame([{
    'TransactionAmount': 7200.0,
    'TransactionHour': 2,
    'IsOnline': 1,
    'CardPresent': 0
}])
 
fraud_prob = model.predict_proba(new_transaction)[0][1]
print(f"\nPredicted Credit Card Fraud Risk: {fraud_prob:.2%}")