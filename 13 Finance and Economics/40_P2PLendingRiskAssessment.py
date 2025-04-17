"""
Project 520: Peer-to-Peer Lending Risk Assessment
Description:
In peer-to-peer (P2P) lending, individuals lend money to other individuals through an online platform. The risk assessment involves predicting the likelihood of a borrower defaulting on the loan based on their financial profile, credit score, loan amount, and loan term. In this project, we will build a predictive model using Logistic Regression to assess the risk of loan default.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 
# 1. Simulate P2P lending data (e.g., borrower financial profile, loan information)
np.random.seed(42)
data = {
    'credit_score': np.random.randint(600, 850, 1000),  # Credit score (600-850)
    'loan_amount': np.random.randint(1000, 50000, 1000),  # Loan amount in USD
    'loan_term': np.random.choice([12, 24, 36, 48, 60], 1000),  # Loan term in months
    'annual_income': np.random.randint(20000, 100000, 1000),  # Borrower's annual income
    'default': np.random.choice([0, 1], 1000)  # 0 = No default, 1 = Default
}
 
df = pd.DataFrame(data)
 
# 2. Define features and target variable
X = df[['credit_score', 'loan_amount', 'loan_term', 'annual_income']]
y = df['default']
 
# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# 6. Make predictions
y_pred = model.predict(X_test)
 
# 7. Evaluate the model
print("P2P Lending Risk Assessment Report:\n")
print(classification_report(y_test, y_pred))
 
# 8. Visualize the predicted default probability (e.g., using probability scores)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of default (class 1)
plt.figure(figsize=(10, 6))
plt.hist(y_prob, bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title("Probability of Loan Default")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()