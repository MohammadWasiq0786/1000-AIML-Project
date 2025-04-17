"""
Project 806. Sales Forecasting Model

Sales forecasting predicts future sales based on historical data. This helps businesses plan inventory, manage resources, and set revenue targets. We'll build a simple time series model using linear regression to forecast monthly sales based on trends.

This model captures a linear trend in monthly sales and extends it to make future predictions. For real-world usage, you can enhance it with seasonal components (using ARIMA, Prophet, or LSTM).
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
 
# Sample monthly sales data
data = {
    'Month': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'Sales': [200, 220, 250, 270, 300, 320, 340, 360, 400, 420, 450, 470]
}
 
df = pd.DataFrame(data)
 
# Convert Month to numeric index for regression
df['MonthIndex'] = np.arange(len(df))
 
# Features (MonthIndex) and target (Sales)
X = df[['MonthIndex']]
y = df['Sales']
 
# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)
 
# Forecast next 6 months
future_months = np.arange(len(df), len(df) + 6).reshape(-1, 1)
future_sales = model.predict(future_months)
 
# Combine past and future data for plotting
forecast_df = pd.DataFrame({
    'Month': pd.date_range(start='2023-01-01', periods=18, freq='M'),
    'Sales': np.concatenate([y, future_sales])
})
 
# Plot actual and forecasted sales
plt.figure(figsize=(10, 5))
plt.plot(forecast_df['Month'], forecast_df['Sales'], marker='o', label='Sales Forecast')
plt.axvline(x=df['Month'].iloc[-1], color='gray', linestyle='--', label='Forecast Start')
plt.title('Monthly Sales Forecast')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
# Show forecasted values
print("Forecasted Sales for Next 6 Months:")
print(pd.DataFrame({'Month': forecast_df['Month'].tail(6), 'ForecastedSales': future_sales.round(2)}))