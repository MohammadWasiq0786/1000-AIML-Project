"""
Project 803. Customer Lifetime Value Prediction

Customer Lifetime Value (CLV) is a key business metric that estimates the total revenue a business can expect from a single customer throughout their relationship. Predicting CLV helps in budgeting, customer acquisition strategy, and personalized marketing. This implementation uses a simplified regression-based approach using RFM (Recency, Frequency, Monetary) features.

This model uses Recency, Frequency, and Monetary values to estimate customer lifetime value. In real applications, more sophisticated models like LTV with time decay, probabilistic models (BG/NBD), or deep learning can be used.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
 
# Sample dataset: each row represents a customer with RFM features and known CLV
data = {
    'Recency': [10, 30, 5, 45, 3, 25, 7, 50],        # days since last purchase
    'Frequency': [15, 5, 25, 2, 30, 6, 18, 1],       # number of purchases
    'Monetary': [200, 100, 400, 50, 500, 120, 300, 30],  # total spend
    'CLV': [1000, 500, 2000, 200, 2500, 600, 1500, 100]  # known lifetime value
}
 
df = pd.DataFrame(data)
 
# Define features and target
X = df[['Recency', 'Frequency', 'Monetary']]
y = df['CLV']
 
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predict CLV on test data
y_pred = model.predict(X_test)
 
# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test set:", round(mse, 2))
 
# Display predictions
predictions = pd.DataFrame({
    'Actual CLV': y_test,
    'Predicted CLV': y_pred
})
print("\nPredicted vs Actual CLV:")
print(predictions)