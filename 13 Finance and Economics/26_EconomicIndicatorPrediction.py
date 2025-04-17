"""
Project 506: Economic Indicator Prediction
Description:
Economic indicator prediction involves forecasting key economic metrics, such as GDP growth, inflation, or unemployment rates, based on historical data and economic trends. These predictions are crucial for decision-making in both public policy and investment strategies. In this project, we will predict an economic indicator (e.g., GDP growth) using historical data and regression models.

For real-world applications:

Use real economic data from sources like FRED (Federal Reserve Economic Data) or World Bank.

Extend the model with additional features like global market trends, political events, or commodity prices.

✅ What It Does:
Simulates economic data for GDP growth, interest rates, and inflation over a period (2000–2020).

Uses linear regression to predict GDP growth based on interest rates and inflation.

Evaluates the model using Mean Squared Error (MSE) to measure prediction accuracy.

Visualizes the relationship between actual and predicted GDP growth.

Key Extensions and Customizations:
Use real-world economic data from sources like FRED, OECD, or World Bank for better accuracy.

Feature engineering: Add other features like unemployment rates, consumer confidence, or global commodity prices to improve the model.

Advanced models: Try more advanced models such as Decision Trees, Random Forests, or XGBoost for better performance.

Time series models: Implement ARIMA, Exponential Smoothing, or LSTM for more accurate time-series forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
 
# 1. Simulate historical economic data (GDP growth and related indicators)
np.random.seed(42)
 
years = np.arange(2000, 2021)
gdp_growth = np.random.normal(2, 0.5, len(years))  # Simulate GDP growth as a normal distribution around 2% with some variability
interest_rate = np.random.normal(3, 1, len(years))  # Simulate interest rates
inflation = np.random.normal(2, 0.8, len(years))  # Simulate inflation rates
 
# Create a DataFrame with simulated data
df = pd.DataFrame({
    'Year': years,
    'GDP_Growth': gdp_growth,
    'Interest_Rate': interest_rate,
    'Inflation': inflation
})
 
# 2. Prepare the data for regression
X = df[['Interest_Rate', 'Inflation']]  # Independent variables: Interest Rate and Inflation
y = df['GDP_Growth']  # Dependent variable: GDP Growth
 
# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 4. Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# 5. Make predictions
y_pred = model.predict(X_test)
 
# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
 
# 7. Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted GDP Growth")
plt.xlabel("Actual GDP Growth")
plt.ylabel("Predicted GDP Growth")
plt.show()
 
# 8. Display the coefficients
print(f"Model Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")