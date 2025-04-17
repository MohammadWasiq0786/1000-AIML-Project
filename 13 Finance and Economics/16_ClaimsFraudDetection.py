"""
Project 496: Claims Fraud Detection
Description:
Claims fraud detection involves identifying fraudulent insurance claims, such as inflated claims, duplicate claims, or misreported information. In this project, we simulate a fraud detection model that uses features like claim amount, claim type, customer history, and location to predict whether a claim is fraudulent or legitimate.

For real-world applications:

Integrate with claims data, customer profiles, and historical fraud data.

Use advanced techniques like anomaly detection, ensemble models, or deep learning.

✅ What It Does:
Simulates claim data with features like claim amount, claim type, customer age, and location.

Uses a Random Forest classifier to predict whether a claim is fraudulent (1) or legitimate (0).

Evaluates model performance using classification metrics like precision, recall, and F1-score.

Visualizes feature importance to understand which features most influence the fraud prediction.

Key Extensions and Customizations:
Use real-world claims data: Integrate with actual insurance claim datasets for more accurate fraud detection.

Advanced techniques: Experiment with XGBoost, Neural Networks, or Anomaly Detection methods for improved performance.

Real-time fraud detection: Build a system that detects fraud in real-time as claims are submitted.
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
 
# 1. Simulate insurance claim data
np.random.seed(42)
data = {
    'claim_amount': np.random.normal(5000, 2000, 1000),  # Claim amount in USD
    'claim_type': np.random.choice(['Accident', 'Fire', 'Theft', 'Health'], 1000),
    'customer_age': np.random.randint(18, 70, 1000),
    'num_previous_claims': np.random.randint(0, 5, 1000),  # Number of previous claims
    'location': np.random.choice(['Urban', 'Suburban', 'Rural'], 1000),
    'fraudulent_claim': np.random.choice([0, 1], 1000)  # 0 = Legitimate, 1 = Fraudulent
}
 
df = pd.DataFrame(data)
 
# 2. Preprocess categorical features
df['claim_type'] = df['claim_type'].map({'Accident': 0, 'Fire': 1, 'Theft': 2, 'Health': 3})
df['location'] = df['location'].map({'Urban': 0, 'Suburban': 1, 'Rural': 2})
 
# 3. Define features and target variable
X = df.drop('fraudulent_claim', axis=1)
y = df['fraudulent_claim']
 
# 4. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 6. Random Forest model for fraud detection
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# 7. Evaluate the model
y_pred = model.predict(X_test)
print("Fraud Detection Model Report:\n")
print(classification_report(y_test, y_pred))
 
# 8. Visualize feature importance (optional)
feature_importances = model.feature_importances_
plt.bar(X.columns, feature_importances)
plt.title("Feature Importance in Fraud Detection")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
 
# 9. Predict fraudulent claims for a new claim
new_claim = np.array([[7000, 2, 45, 1, 0]])  # Example: Claim amount = 7000, Fire claim, customer age = 45, 1 previous claim, Urban location
new_claim_scaled = scaler.transform(new_claim)
predicted_fraud = model.predict(new_claim_scaled)
print(f"\nPredicted Claim Fraud: {'Fraudulent' if predicted_fraud[0] == 1 else 'Legitimate'}")