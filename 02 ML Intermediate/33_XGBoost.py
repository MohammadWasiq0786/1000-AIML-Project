"""
Project 73. XGBoost implementation for tabular data
Description:
XGBoost (Extreme Gradient Boosting) is a high-performance, scalable machine learning algorithm optimized for accuracy and speed. In this project, we apply XGBoost to a simulated tabular dataset to perform classification, showcasing its power on structured data.

About:
🌲 What This Project Demonstrates:
Applies XGBoost to high-dimensional tabular data

Uses binary classification with evaluation metrics

Leverages gradient boosting for accuracy and speed
"""

# Install if not already: pip install xgboost
 
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
 
# Create synthetic tabular data
X, y = make_classification(n_samples=1000, n_features=15, n_informative=10,
                           n_classes=2, random_state=42)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Convert to XGBoost DMatrix (optional but optimized for speed)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
 
# XGBoost parameters for binary classification
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,
    'eta': 0.1,
    'seed': 42
}
 
# Train the model
model = xgb.train(params, dtrain, num_boost_round=100)
 
# Predict
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)
 
# Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - XGBoost Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()