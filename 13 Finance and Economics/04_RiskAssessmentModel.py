"""
Project 484: Risk Assessment Model
Description:
Risk assessment is crucial in finance for evaluating potential investment risks or the creditworthiness of borrowers. In this project, we simulate a risk scoring system for credit applications, where we use logistic regression to predict the likelihood of a borrower defaulting on a loan based on their financial features.

In a real-world system, you can use:

Credit Bureau data, FICO scores, loan history datasets

✅ What It Does:
Simulates borrower data including credit score, income, loan amount, debt-to-income ratio, and employment status.

Logistic regression is used to predict whether a borrower will default on a loan.

Evaluates the model's performance using precision, recall, and F1-score metrics.

Scales numerical features and makes predictions for new applicants.

Key Extensions and Customizations:
Use real-world financial data: Integrate datasets like credit history, loan repayment records, or customer data for more accurate predictions.

Advanced modeling: Implement more sophisticated models like random forests, XGBoost, or neural networks for better performance.

Model explainability: Use techniques like SHAP or LIME to explain model predictions to end-users.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
# 1. Simulated borrower data
np.random.seed(42)
data = {
    'credit_score': np.random.randint(300, 850, 1000),
    'annual_income': np.random.normal(50000, 15000, 1000),  # In USD
    'loan_amount': np.random.normal(20000, 5000, 1000),  # In USD
    'debt_to_income_ratio': np.random.uniform(0.1, 0.5, 1000),
    'previous_default': np.random.choice([0, 1], 1000),  # 0 = no, 1 = yes
    'employment_status': np.random.choice(['Employed', 'Unemployed'], 1000),
    'loan_default': np.random.choice([0, 1], 1000)  # 0 = No Default, 1 = Default
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
df['employment_status'] = df['employment_status'].map({'Employed': 1, 'Unemployed': 0})
X = df.drop('loan_default', axis=1)
y = df['loan_default']
 
# 3. Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['credit_score', 'annual_income', 'loan_amount', 'debt_to_income_ratio']])
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# 6. Evaluate the model
y_pred = model.predict(X_test)
print("Loan Default Prediction Report:\n")
print(classification_report(y_test, y_pred))
 
# 7. Predict risk for a new borrower
new_borrower = np.array([[750, 60000, 15000, 0.2]])  # Example borrower data
new_borrower_scaled = scaler.transform(new_borrower)
predicted_risk = model.predict(new_borrower_scaled)
print(f"\nPredicted loan default risk for the new borrower: {'Default' if predicted_risk[0] == 1 else 'No Default'}")