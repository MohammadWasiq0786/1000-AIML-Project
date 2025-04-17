"""
Project 519: Corporate Earnings Prediction
Description:
Corporate earnings prediction involves forecasting a company's future earnings per share (EPS) based on historical financial data, market conditions, and economic indicators. In this project, we will use regression models to predict the earnings of a company (e.g., Apple) based on its financial ratios and other relevant features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 
# 1. Simulate financial data for a company (e.g., revenue, expenses, market sentiment)
np.random.seed(42)
data = {
    'revenue': np.random.normal(50000, 10000, 1000),  # Revenue in USD
    'expenses': np.random.normal(30000, 8000, 1000),  # Expenses in USD
    'market_sentiment': np.random.normal(0.5, 0.1, 1000),  # Market sentiment index (0 to 1)
    'earnings_per_share': np.random.normal(5, 2, 1000)  # Earnings per share in USD
}
 
df = pd.DataFrame(data)
 
# 2. Define features and target variable
X = df[['revenue', 'expenses', 'market_sentiment']]
y = df['earnings_per_share']
 
# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# 6. Make predictions
y_pred = model.predict(X_test)
 
# 7. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: ${mae:.2f}")
 
# 8. Plot actual vs predicted earnings per share
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted Earnings Per Share")
plt.xlabel("Actual Earnings Per Share")
plt.ylabel("Predicted Earnings Per Share")
plt.show()