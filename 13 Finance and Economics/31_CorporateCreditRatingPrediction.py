"""
Project 511: Corporate Credit Rating Prediction
Description:
Corporate credit rating prediction is essential for financial institutions to assess the creditworthiness of companies. Credit ratings indicate the likelihood that a company will default on its debt obligations. In this project, we will use financial features such as debt levels, profitability, and liquidity ratios to predict a company's credit rating (e.g., AAA, AA, A, BBB, etc.) using machine learning models.

🧪 Python Implementation (Corporate Credit Rating Prediction with Logistic Regression)
For real-world applications:

Use real credit rating datasets (e.g., Moody's or S&P Global data) for training the model.

You can extend the model by integrating macroeconomic factors or market data to improve prediction accuracy.

About:
✅ What It Does:
Simulates financial data for companies, including features like debt-to-equity, return on assets, current ratio, and interest coverage ratio.

Uses Logistic Regression to classify the company's credit rating into categories such as AAA, AA, A, and BBB.

Evaluates the model using classification metrics such as precision, recall, and F1-score.

Visualizes the feature importance (coefficients of the logistic regression) to understand which financial ratios most influence the credit rating prediction.

Key Extensions and Customizations:
Real-world data: Use actual credit rating datasets from Moody's, S&P Global, or other financial institutions.

Advanced models: Use models like Random Forest, XGBoost, or Neural Networks to improve prediction accuracy.

Macroeconomic data: Integrate broader economic indicators like interest rates, GDP growth, or inflation to enhance the model's forecasting capabilities.

Credit risk analysis: Extend this model to assess credit risk by incorporating additional features such as historical defaults or market volatility.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 
# 1. Simulate financial data for companies (debt-to-equity, return on assets, liquidity, etc.)
np.random.seed(42)
data = {
    'debt_to_equity': np.random.normal(1.5, 0.5, 1000),  # Debt-to-equity ratio
    'return_on_assets': np.random.normal(0.05, 0.02, 1000),  # Return on assets
    'current_ratio': np.random.normal(1.2, 0.3, 1000),  # Current ratio (liquidity)
    'interest_coverage': np.random.normal(3, 1.5, 1000),  # Interest coverage ratio
    'credit_rating': np.random.choice([1, 2, 3, 4], 1000)  # 1 = AAA, 2 = AA, 3 = A, 4 = BBB
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
X = df.drop('credit_rating', axis=1)
y = df['credit_rating']
 
# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Logistic Regression model for credit rating prediction
model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)
 
# 6. Make predictions
y_pred = model.predict(X_test)
 
# 7. Evaluate the model
print("Corporate Credit Rating Prediction Report:\n")
print(classification_report(y_test, y_pred))
 
# 8. Plot the feature importance (coefficients of the logistic regression)
coefficients = model.coef_[0]
features = X.columns
 
plt.figure(figsize=(10, 6))
plt.bar(features, coefficients)
plt.title("Feature Importance in Corporate Credit Rating Prediction")
plt.xlabel("Financial Ratios")
plt.ylabel("Coefficient Value")
plt.show()
 
# 9. Predict credit rating for a new company based on financial ratios
new_data = np.array([[2.0, 0.06, 1.0, 2]])  # Example: debt_to_equity=2.0, return_on_assets=6%, current_ratio=1.0, interest_coverage=2
new_data_scaled = scaler.transform(new_data)
predicted_rating = model.predict(new_data_scaled)
rating_map = {1: 'AAA', 2: 'AA', 3: 'A', 4: 'BBB'}
print(f"\nPredicted Credit Rating: {rating_map[predicted_rating[0]]}")