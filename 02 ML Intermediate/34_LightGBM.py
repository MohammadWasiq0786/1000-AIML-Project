"""
Project 74. LightGBM model for large datasets
Description:
LightGBM (Light Gradient Boosting Machine) is a highly efficient, scalable boosting framework especially suited for large datasets. It uses histogram-based learning and leaf-wise tree growth for speed and performance. In this project, we apply LightGBM to a large simulated tabular dataset for classification.

About:
⚡ What This Project Demonstrates:
Handles large-scale data efficiently with LightGBM

Trains a fast and accurate binary classifier

Evaluates using classification metrics and confusion matrix

Supports early stopping for optimized performance
"""

# Install if not already: pip install lightgbm
 
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
 
# Generate large synthetic dataset
X, y = make_classification(n_samples=50000, n_features=30, n_informative=20,
                           n_classes=2, random_state=42)
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Create LightGBM dataset objects
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
 
# Define parameters for binary classification
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}
 
# Train LightGBM model
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, early_stopping_rounds=10)
 
# Predict
y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_prob > 0.5).astype(int)
 
# Evaluate performance
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - LightGBM Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()