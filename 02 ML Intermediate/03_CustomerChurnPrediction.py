"""
Project 43. Customer churn prediction model
Description:
Customer churn prediction aims to identify customers likely to leave (churn) a service or subscription. By analyzing customer behavior, businesses can proactively retain at-risk users. In this project, we use a logistic regression model to predict churn from customer data using scikit-learn.

About:
🧠 What This Project Demonstrates:
Predicts churn (Yes/No) using behavioral and contract features

Preprocesses both categorical and numerical data

Evaluates performance with classification metrics and confusion matrix
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
# Sample simulated customer churn data
data = {
    'Age': [25, 45, 39, 50, 30, 28, 55, 23, 40, 35],
    'MonthlyCharges': [50, 80, 70, 100, 60, 55, 90, 45, 75, 65],
    'Contract': ['Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Two year',
                 'Month-to-month', 'One year', 'Month-to-month', 'Two year', 'One year'],
    'Tenure': [1, 24, 5, 18, 36, 3, 20, 2, 28, 12],
    'Churn': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No']
}
 
df = pd.DataFrame(data)
 
# Encode categorical variables
le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])  # Month-to-month: 0, One year: 1, Two year: 2
df['Churn'] = le.fit_transform(df['Churn'])        # Yes: 1, No: 0
 
# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']
 
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
 
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix - Customer Churn")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()