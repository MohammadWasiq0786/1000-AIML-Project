"""
Project 514: Real Estate Price Prediction
Description:
Real estate price prediction involves forecasting the price of properties based on features like location, size, number of rooms, and age of the property. In this project, we will use a regression model to predict real estate prices based on these features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 
# 1. Simulate real estate data (e.g., property size, number of rooms, etc.)
np.random.seed(42)
data = {
    'size': np.random.normal(1500, 500, 1000),  # Square footage
    'num_rooms': np.random.randint(2, 6, 1000),  # Number of rooms
    'age': np.random.randint(1, 50, 1000),  # Age of the property in years
    'price': np.random.normal(300000, 50000, 1000)  # Price of the property in USD
}
 
df = pd.DataFrame(data)
 
# 2. Define features and target variable
X = df[['size', 'num_rooms', 'age']]
y = df['price']
 
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
 
# 8. Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted Real Estate Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()