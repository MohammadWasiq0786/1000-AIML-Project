"""
Project 509: Financial Crisis Early Warning System
Description:
A financial crisis early warning system aims to detect signs of an impending financial crisis by analyzing macroeconomic indicators such as interest rates, inflation, unemployment rates, and stock market performance. In this project, we will build a simple machine learning model to predict the likelihood of a financial crisis based on these indicators.

For real-world systems:

Use historical data from sources like FRED (Federal Reserve Economic Data) or OECD to train and validate the model.

Extend this model by incorporating more macroeconomic indicators or financial market data for more robust predictions.

✅ What It Does:
Simulates macroeconomic data (interest rates, inflation, unemployment, and stock market performance).

Uses Logistic Regression to classify whether a financial crisis is likely based on the given economic conditions.

Evaluates the model’s performance using classification metrics like precision, recall, and F1-score.

Plots the feature importance (coefficients of the logistic regression) to understand which macroeconomic indicators most influence crisis predictions.

Key Extensions and Customizations:
Real-world data: Replace the simulated data with actual economic indicators from sources like FRED, OECD, or World Bank.

Advanced models: Use more sophisticated models such as Random Forest, SVMs, or XGBoost for better predictive power.

Additional indicators: Include housing market data, consumer sentiment, or global economic factors to improve predictions.

Scenario analysis: Perform scenario analysis to see how different shocks to key economic variables affect the likelihood of a financial crisis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 
# 1. Simulate macroeconomic data (e.g., interest rates, inflation, unemployment rates)
np.random.seed(42)
data = {
    'interest_rate': np.random.normal(5, 1.5, 1000),  # Simulated interest rate
    'inflation': np.random.normal(2, 0.8, 1000),  # Simulated inflation rate
    'unemployment': np.random.normal(4, 1, 1000),  # Simulated unemployment rate
    'stock_market_performance': np.random.normal(0, 0.2, 1000),  # Simulated stock market returns (daily)
    'financial_crisis': np.random.choice([0, 1], 1000)  # 0 = No crisis, 1 = Crisis
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
X = df.drop('financial_crisis', axis=1)
y = df['financial_crisis']
 
# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Logistic Regression model to predict financial crises
model = LogisticRegression()
model.fit(X_train, y_train)
 
# 6. Make predictions
y_pred = model.predict(X_test)
 
# 7. Evaluate the model
print("Financial Crisis Prediction Report:\n")
print(classification_report(y_test, y_pred))
 
# 8. Plot the feature importance (coefficients of the logistic regression)
coefficients = model.coef_[0]
features = X.columns
 
plt.figure(figsize=(10, 6))
plt.bar(features, coefficients)
plt.title("Feature Importance in Financial Crisis Prediction")
plt.xlabel("Economic Indicators")
plt.ylabel("Coefficient Value")
plt.show()
 
# 9. Predict financial crisis for a new set of economic conditions
new_data = np.array([[6, 3.5, 5, 0.01]])  # Example: interest_rate=6%, inflation=3.5%, unemployment=5%, stock_performance=0.01%
new_data_scaled = scaler.transform(new_data)
predicted_crisis = model.predict(new_data_scaled)
print(f"\nPredicted Financial Crisis: {'Yes' if predicted_crisis[0] == 1 else 'No'}")