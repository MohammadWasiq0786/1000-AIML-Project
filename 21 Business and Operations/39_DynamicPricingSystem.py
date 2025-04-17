"""
Project 839. Dynamic Pricing System

A dynamic pricing system adjusts product prices in real time based on factors like demand, inventory, competition, and time. This project simulates pricing decisions using a regression model to predict optimal price that maximizes revenue based on input features.

This simple model suggests prices based on demand, inventory, and competitor pricing. In production, dynamic pricing systems may also include:

Time-based features (e.g., hour, day of week, season)

Machine learning models (e.g., XGBoost, reinforcement learning)

Revenue simulations or constraint-based optimization
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
 
# Simulated dataset of product pricing context
data = {
    'DemandLevel': [100, 80, 120, 90, 70, 110, 130, 60],  # units demanded
    'StockLevel': [50, 70, 30, 60, 80, 40, 20, 90],       # current inventory
    'CompetitorPrice': [19, 21, 18, 20, 22, 17, 16, 23],  # market price
    'OptimalPrice': [20, 21, 19, 20.5, 21.5, 18, 17, 22]  # target price to predict
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df[['DemandLevel', 'StockLevel', 'CompetitorPrice']]
y = df['OptimalPrice']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predict optimal price for a new context
new_context = pd.DataFrame([{
    'DemandLevel': 115,
    'StockLevel': 35,
    'CompetitorPrice': 18.5
}])
 
predicted_price = model.predict(new_context)[0]
print(f"Predicted Optimal Price: ${predicted_price:.2f}")