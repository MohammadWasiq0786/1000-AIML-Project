"""
Project 883. Phishing Detection System

A phishing detection system identifies malicious websites or emails that attempt to steal sensitive information (e.g., login credentials, credit card numbers). In this project, we simulate URL-based features and use a classification model to detect phishing attempts.

This rule-based model identifies phishing attempts based on URL features. For production systems, enhance it using:

Text analysis of email content or HTML source

Domain name reputation APIs

Neural models (e.g., LSTM or transformers) for sequence-aware detection
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# Simulated dataset with URL-based features
data = {
    'URLLength': [50, 120, 35, 95, 80, 25, 150, 45],
    'NumDots': [1, 5, 1, 4, 3, 1, 6, 2],
    'HasHTTPS': [1, 0, 1, 0, 1, 1, 0, 1],
    'NumSpecialChars': [3, 10, 2, 9, 6, 2, 12, 4],
    'Phishing': [0, 1, 0, 1, 0, 0, 1, 0]  # 1 = phishing, 0 = legitimate
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df.drop('Phishing', axis=1)
y = df['Phishing']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train the phishing classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Evaluate on test set
y_pred = model.predict(X_test)
print("Phishing Detection Classification Report:")
print(classification_report(y_test, y_pred))
 
# Predict phishing likelihood for a new URL
new_url = pd.DataFrame([{
    'URLLength': 110,
    'NumDots': 4,
    'HasHTTPS': 0,
    'NumSpecialChars': 9
}])
 
phishing_prob = model.predict_proba(new_url)[0][1]
print(f"\nPredicted Phishing Risk: {phishing_prob:.2%}")