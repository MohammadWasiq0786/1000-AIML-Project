"""
Project 486: Loan Default Prediction
Description:
Loan default prediction helps financial institutions determine whether a borrower is likely to default on a loan. The goal is to use various features (e.g., credit score, income, loan amount, and financial behavior) to predict the likelihood of default. In this project, we will use a Random Forest Classifier to predict whether a borrower will default based on their financial information.

For real-world applications:

Use datasets from loan providers, credit bureaus, or banking systems.

✅ What It Does:
Simulates loan data (credit score, income, loan amount, debt-to-income ratio) and predicts loan default.

Uses Random Forest Classifier to build a model that predicts whether a loan will default.

Evaluates the model using classification metrics like precision, recall, and F1-score.

Scales the features to improve model performance.

Key Extensions and Customizations:
Use real-world loan data to enhance the accuracy and reliability of the model.

Incorporate more features like employment history, financial assets, and loan type.

Model improvement: Experiment with Gradient Boosting models (e.g., XGBoost, LightGBM) for better predictive power.

Add interpretability: Use tools like SHAP or LIME to explain the model's decisions and provide transparency in loan approval processes.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
 
# 1. Simulated loan data (features: credit score, income, loan amount, history)
np.random.seed(42)
data = {
    'credit_score': np.random.randint(300, 850, 1000),
    'annual_income': np.random.normal(60000, 15000, 1000),  # In USD
    'loan_amount': np.random.normal(25000, 7000, 1000),  # In USD
    'debt_to_income_ratio': np.random.uniform(0.05, 0.45, 1000),
    'previous_default': np.random.choice([0, 1], 1000),  # 0 = no, 1 = yes
    'loan_default': np.random.choice([0, 1], 1000)  # 0 = No Default, 1 = Default
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
X = df.drop('loan_default', axis=1)
y = df['loan_default']
 
# 3. Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['credit_score', 'annual_income', 'loan_amount', 'debt_to_income_ratio']])
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Random Forest model for prediction
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# 6. Evaluate the model
y_pred = model.predict(X_test)
print("Loan Default Prediction Report:\n")
print(classification_report(y_test, y_pred))
 
# 7. Predict loan default for a new borrower
new_borrower = np.array([[700, 65000, 20000, 0.2]])  # Example borrower data
new_borrower_scaled = scaler.transform(new_borrower)
predicted_default = model.predict(new_borrower_scaled)
print(f"\nPredicted Loan Default: {'Default' if predicted_default[0] == 1 else 'No Default'}")