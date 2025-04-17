"""
Project 485: Credit Scoring System
Description:
A Credit Scoring System evaluates a borrower’s creditworthiness by analyzing various financial features such as income, credit history, loan amount, etc. This system is similar to the FICO score and is crucial for financial institutions to determine the risk of lending. In this project, we will simulate a credit scoring model using decision trees to predict whether a borrower qualifies for a loan based on their financial profile.

For real-world data:

You can use datasets from FICO, credit bureaus, or bank customer data.

✅ What It Does:
Simulates a credit scoring model using financial features like credit score, income, loan amount, and debt-to-income ratio.

Uses a decision tree to predict whether a borrower will be approved for credit based on the input features.

Evaluates model performance using classification metrics like precision, recall, and F1-score.

Scales the features for improved model performance and accuracy.

Key Extensions and Customizations:
Use real-world credit data to improve the model's accuracy and reliability.

Incorporate more features such as credit utilization, loan history, employment duration, and financial dependents.

Model optimization: Experiment with Random Forests, XGBoost, or Neural Networks for better performance.

Add interpretability: Use techniques like SHAP or LIME to explain how the model makes credit approval decisions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
 
# 1. Simulated borrower data
np.random.seed(42)
data = {
    'credit_score': np.random.randint(300, 850, 1000),
    'annual_income': np.random.normal(60000, 15000, 1000),  # In USD
    'loan_amount': np.random.normal(25000, 7000, 1000),  # In USD
    'debt_to_income_ratio': np.random.uniform(0.05, 0.45, 1000),
    'previous_default': np.random.choice([0, 1], 1000),  # 0 = no, 1 = yes
    'employment_status': np.random.choice(['Employed', 'Unemployed'], 1000),
    'credit_approved': np.random.choice([0, 1], 1000)  # 0 = not approved, 1 = approved
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
df['employment_status'] = df['employment_status'].map({'Employed': 1, 'Unemployed': 0})
X = df.drop('credit_approved', axis=1)
y = df['credit_approved']
 
# 3. Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['credit_score', 'annual_income', 'loan_amount', 'debt_to_income_ratio']])
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
 
# 6. Evaluate the model
y_pred = model.predict(X_test)
print("Credit Scoring System Report:\n")
print(classification_report(y_test, y_pred))
 
# 7. Predict credit approval for a new borrower
new_borrower = np.array([[700, 65000, 20000, 0.2]])  # Example borrower data
new_borrower_scaled = scaler.transform(new_borrower)
predicted_approval = model.predict(new_borrower_scaled)
print(f"\nPredicted Credit Approval: {'Approved' if predicted_approval[0] == 1 else 'Not Approved'}")