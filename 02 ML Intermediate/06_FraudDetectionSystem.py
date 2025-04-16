"""
Project 46. Fraud detection system
Description:
Fraud detection systems are designed to identify unusual behavior in financial transactions that may indicate fraud. In this project, we implement a binary classifier using logistic regression on an imbalanced dataset (simulated) to predict fraudulent transactions. We also apply resampling techniques to handle class imbalance and evaluate using appropriate metrics.

About:
🛡️ What This Project Demonstrates:
Detects rare events (fraud) using logistic regression

Handles imbalanced data via oversampling

Evaluates using precision, recall, ROC-AUC, and confusion matrix
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
 
# Simulated transaction dataset
np.random.seed(42)
n_samples = 1000
fraud_ratio = 0.05
 
# Generate features
data = {
    'TransactionAmount': np.random.exponential(scale=100, size=n_samples),
    'TransactionTime': np.random.randint(0, 24, size=n_samples),
    'IsForeign': np.random.randint(0, 2, size=n_samples),
    'IsHighRiskCountry': np.random.randint(0, 2, size=n_samples),
    'Fraud': np.zeros(n_samples, dtype=int)
}
 
# Randomly assign fraud labels
fraud_indices = np.random.choice(n_samples, int(fraud_ratio * n_samples), replace=False)
for idx in fraud_indices:
    data['Fraud'][idx] = 1
 
df = pd.DataFrame(data)
 
# Separate majority and minority classes
df_majority = df[df.Fraud == 0]
df_minority = df[df.Fraud == 1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    
                                 n_samples=len(df_majority),     
                                 random_state=42) 
 
# Combine datasets
df_balanced = pd.concat([df_majority, df_minority_upsampled])
 
# Features and target
X = df_balanced.drop('Fraud', axis=1)
y = df_balanced['Fraud']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
 
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title("Confusion Matrix - Fraud Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
 
# ROC-AUC Score
roc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_score:.2f}")