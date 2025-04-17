"""
Project 495: Insurance Pricing Optimization
Description:
Insurance pricing optimization is the process of determining the best pricing strategy for insurance policies based on customer risk profiles, coverage amounts, and other relevant factors. In this project, we'll simulate an insurance pricing model that uses features like age, health status, coverage type, and previous claims to predict the appropriate insurance premium.

For real-world applications:

Use customer data, such as age, health status, claim history, and coverage types, to determine pricing.

Extend this project to include market conditions, competitor pricing, and regulatory factors.

✅ What It Does:
Simulates insurance data with features like age, health status, coverage amount, and previous claims.

Uses Linear Regression to predict the insurance premium based on these features.

Evaluates the model with Mean Absolute Error (MAE) to assess prediction accuracy.

Visualizes the predicted premiums vs. actual premiums with a scatter plot.

Key Extensions and Customizations:
Non-linear models: Use Random Forests or Gradient Boosting Machines to capture non-linear relationships between features and premiums.

Market data: Integrate market conditions or competitor pricing to adjust premiums dynamically.

Dynamic pricing: Implement real-time data updates (e.g., incorporating new claims, market changes) for automated pricing adjustments.
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
 
# 1. Simulate insurance data
np.random.seed(42)
data = {
    'age': np.random.randint(18, 70, 1000),
    'health_status': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], 1000),
    'coverage_amount': np.random.normal(100000, 25000, 1000),  # Coverage in USD
    'previous_claims': np.random.randint(0, 5, 1000),  # Number of claims
    'insurance_premium': np.random.normal(1200, 300, 1000)  # Premium in USD
}
 
df = pd.DataFrame(data)
 
# 2. Preprocess categorical features (e.g., health_status)
df['health_status'] = df['health_status'].map({'Excellent': 0, 'Good': 1, 'Fair': 2, 'Poor': 3})
 
# 3. Define features and target variable
X = df.drop('insurance_premium', axis=1)
y = df['insurance_premium']
 
# 4. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 6. Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# 7. Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error for Insurance Pricing Model: {mae:.2f}")
 
# 8. Plot the actual vs predicted insurance premiums
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title("Actual vs Predicted Insurance Premiums")
plt.xlabel("Actual Premiums (USD)")
plt.ylabel("Predicted Premiums (USD)")
plt.show()