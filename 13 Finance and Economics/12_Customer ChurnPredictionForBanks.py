"""
Project 492: Customer Churn Prediction for Banks
Description:
Customer churn prediction helps banks identify customers who are likely to leave the bank (i.e., close their accounts) in the near future. By predicting churn, banks can proactively take action to retain high-risk customers. In this project, we will use logistic regression to predict whether a customer will churn based on features like account balance, number of transactions, and customer demographics.

For real-world applications:

Use customer data such as account tenure, transaction history, demographics, service usage, etc.

✅ What It Does:
Simulates customer data (account balance, age, number of transactions, years with bank) and predicts whether a customer will churn.

Uses Logistic Regression to classify customers into churn or non-churn categories.

Evaluates model performance with classification metrics such as precision, recall, and F1-score.

Predicts churn for new customers based on their features.

Key Extensions and Customizations:
Use real-world customer data: Integrate datasets from banks, telecoms, or e-commerce platforms for more accurate predictions.

Add more features: Include additional features like customer support interactions, complaint history, or product/service usage.

Model improvement: Implement more advanced models such as Random Forest, XGBoost, or Neural Networks for better accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
# 1. Simulate customer data
np.random.seed(42)
data = {
    'account_balance': np.random.normal(5000, 1500, 1000),  # USD
    'age': np.random.randint(18, 70, 1000),
    'num_transactions': np.random.randint(1, 20, 1000),
    'years_with_bank': np.random.randint(1, 20, 1000),
    'churn': np.random.choice([0, 1], 1000)  # 0 = stay, 1 = churn
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
X = df.drop('churn', axis=1)
y = df['churn']
 
# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# 6. Evaluate the model
y_pred = model.predict(X_test)
print("Customer Churn Prediction Report:\n")
print(classification_report(y_test, y_pred))
 
# 7. Predict churn for a new customer
new_customer = np.array([[4500, 35, 10, 5]])  # Example customer: balance = 4500, age = 35, 10 transactions, 5 years with the bank
new_customer_scaled = scaler.transform(new_customer)
predicted_churn = model.predict(new_customer_scaled)
print(f"\nPredicted Churn: {'Churn' if predicted_churn[0] == 1 else 'No Churn'}")