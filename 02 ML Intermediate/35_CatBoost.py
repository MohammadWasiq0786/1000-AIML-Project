"""
Project 75. CatBoost algorithm implementation
Description:
CatBoost (Categorical Boosting) is a powerful gradient boosting library developed by Yandex, known for handling categorical features natively, delivering high performance with minimal preprocessing. In this project, we implement a CatBoost classifier on tabular data with categorical features and evaluate its performance.

About:
🐱 What This Project Demonstrates:
Implements CatBoost with mixed data types (categorical + numerical)

Requires minimal preprocessing for categorical features

Provides accurate, efficient gradient boosting classification
"""

# Install if not already: pip install catboost
 
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
# Simulated dataset with categorical and numerical features
data = {
    'Education': ['Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'PhD', 'Bachelors'],
    'Experience': [2, 5, 10, 3, 7, 12, 4, 6, 11, 1],
    'Department': ['HR', 'IT', 'Finance', 'Finance', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance'],
    'Target': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
}
 
df = pd.DataFrame(data)
 
# Define features and target
X = df.drop('Target', axis=1)
y = df['Target']
 
# Identify categorical feature indices
cat_features = [0, 2]  # column indices of 'Education' and 'Department'
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Initialize and train CatBoost model
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, verbose=0)
model.fit(X_train, y_train, cat_features=cat_features)
 
# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix - CatBoost Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()