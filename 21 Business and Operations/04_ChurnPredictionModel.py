"""
Project 804. Churn Prediction Model

Churn prediction helps businesses identify customers who are likely to stop using their services or products. By detecting churn risks early, companies can take action to retain those customers. In this implementation, we’ll use a logistic regression model on customer behavior data to predict whether a customer will churn (1) or not (0).

This model predicts customer churn based on their billing and tenure history. In real-world applications, more features (e.g., support calls, service usage, demographics) and advanced models like Random Forest or XGBoost can boost performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
 
# Sample dataset: each row represents a customer with usage features and churn status
data = {
    'Tenure': [1, 24, 12, 5, 36, 6, 30, 4],  # months with the company
    'MonthlyCharges': [70, 30, 50, 65, 25, 55, 20, 75],  # monthly bill
    'TotalCharges': [70, 720, 600, 325, 900, 330, 600, 300],  # total spend
    'Churn': [1, 0, 0, 1, 0, 1, 0, 1]  # 1 = churned, 0 = retained
}
 
df = pd.DataFrame(data)
 
# Define features and target
X = df[['Tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Predict on test set
y_pred = model.predict(X_test)
 
# Evaluate model performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
 
print("\nClassification Report:")
print(classification_report(y_test, y_pred))