"""
Project 57. Lead scoring model
Description:
A lead scoring model predicts how likely a sales lead is to convert into a paying customer. This helps prioritize leads and improve marketing efficiency. In this project, we build a classification model to score leads using features like demographics, behavior, and engagement level.

About:
🎯 What This Project Demonstrates:
Builds a lead scoring classifier using marketing and behavioral features

Handles categorical encoding and scaling

Uses a Random Forest for robust prediction and interpretability

Evaluates model with classification report and confusion matrix
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
# Simulated lead data
data = {
    'LeadSource': ['Email', 'Website', 'Referral', 'Email', 'Ad', 'Website', 'Ad', 'Referral', 'Website', 'Email'],
    'Age': [25, 34, 28, 45, 39, 22, 31, 29, 35, 27],
    'PageViews': [5, 10, 3, 20, 15, 2, 8, 4, 12, 6],
    'TimeOnSite': [300, 900, 200, 1500, 800, 180, 700, 240, 1100, 350],
    'Converted': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]  # 1 = Lead converted, 0 = did not
}
 
df = pd.DataFrame(data)
 
# Encode categorical data
le = LabelEncoder()
df['LeadSource'] = le.fit_transform(df['LeadSource'])  # Convert to numeric
 
# Split features and target
X = df.drop('Converted', axis=1)
y = df['Converted']
 
# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
 
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Not Converted', 'Converted'], yticklabels=['Not Converted', 'Converted'])
plt.title("Confusion Matrix - Lead Scoring Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()