"""
Project 52. Income prediction model
Description:
An income prediction model estimates a person’s income based on features such as age, education, occupation, and work hours. This can be used for socioeconomic analysis or targeted marketing. In this project, we build a regression model using scikit-learn to predict annual income based on synthetic data.

About:
💰 What This Project Demonstrates:
Uses regression modeling to predict income from demographic/work data

Encodes categorical features and standardizes input

Evaluates performance with MSE and R², and visualizes results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
 
# Simulated demographic dataset
data = {
    'Age': [25, 32, 40, 28, 45, 50, 22, 36, 30, 48],
    'Education': ['Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'PhD', 'High School', 'Masters', 'Bachelors', 'PhD'],
    'Occupation': ['Engineer', 'Scientist', 'Professor', 'Engineer', 'Scientist', 'Professor', 'Technician', 'Scientist', 'Technician', 'Professor'],
    'HoursPerWeek': [40, 50, 45, 42, 55, 60, 38, 48, 40, 52],
    'Income': [55000, 72000, 95000, 58000, 80000, 98000, 40000, 76000, 47000, 97000]
}
 
df = pd.DataFrame(data)
 
# Encode categorical variables
le_edu = LabelEncoder()
le_occ = LabelEncoder()
df['Education'] = le_edu.fit_transform(df['Education'])
df['Occupation'] = le_occ.fit_transform(df['Occupation'])
 
# Features and target
X = df.drop('Income', axis=1)
y = df['Income']
 
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
 
# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
 
print("Predicted Incomes:", np.round(y_pred, 2))
print("Mean Squared Error:", round(mse, 2))
print("R² Score:", round(r2, 2))
 
# Visualize actual vs predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title("Actual vs Predicted Income")
plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.grid(True)
plt.tight_layout()
plt.show()