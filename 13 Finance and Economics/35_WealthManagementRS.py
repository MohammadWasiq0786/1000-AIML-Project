"""
Project 515: Wealth Management Recommendation System
Description:
A wealth management recommendation system helps individuals manage their finances by providing personalized advice on investments, savings, and asset allocation. In this project, we will build a simple recommendation system that suggests personalized investment portfolios based on user risk preferences and financial goals.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
# 1. Simulate user data (e.g., risk tolerance, income, age)
np.random.seed(42)
data = {
    'age': np.random.randint(18, 70, 1000),
    'income': np.random.randint(30000, 150000, 1000),
    'risk_tolerance': np.random.choice(['Low', 'Medium', 'High'], 1000),
    'investment_strategy': np.random.choice(['Conservative', 'Balanced', 'Aggressive'], 1000)
}
 
df = pd.DataFrame(data)
 
# 2. Encode categorical variables
df['risk_tolerance'] = df['risk_tolerance'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['investment_strategy'] = df['investment_strategy'].map({'Conservative': 0, 'Balanced': 1, 'Aggressive': 2})
 
# 3. Define features and target variable
X = df[['age', 'income', 'risk_tolerance']]
y = df['investment_strategy']
 
# 4. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 6. Train the K-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
 
# 7. Make predictions for a new user
new_user = np.array([[35, 50000, 1]])  # Example: age=35, income=50k, risk_tolerance=Medium
new_user_scaled = scaler.transform(new_user)
predicted_strategy = model.predict(new_user_scaled)
 
strategies = {0: 'Conservative', 1: 'Balanced', 2: 'Aggressive'}
print(f"Recommended Investment Strategy: {strategies[predicted_strategy[0]]}")