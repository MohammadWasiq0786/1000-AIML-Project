"""
Project 475. Healthcare fraud detection
Description:
Healthcare fraud detection aims to identify anomalies or suspicious patterns in insurance claims, provider behavior, or billing data. Fraud examples include overbilling, phantom claims, or unbundling services. In this project, we simulate claims data and use unsupervised anomaly detection to flag potential fraud.

About:
✅ What It Does:
Simulates insurance claim data with both normal and fraudulent patterns.

Uses Isolation Forest, a popular anomaly detection algorithm.

Flags outliers based on abnormal claim amount, number of services, etc.

Extendable to:

Use claim diagnosis codes or temporal patterns

Train supervised models if labeled fraud data exists

Build a real-time fraud detection pipeline

Real-world data:

Use datasets like CMS Medicare Claims, MIMIC-Billing, or synthetic EHRs
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
 
# 1. Simulate claims data
np.random.seed(42)
n_normal = 950
n_fraud = 50
 
# Normal claims
normal_claims = pd.DataFrame({
    "claim_amount": np.random.normal(500, 100, n_normal),
    "num_services": np.random.poisson(3, n_normal),
    "provider_rating": np.random.uniform(3.5, 5.0, n_normal),
    "days_in_hospital": np.random.poisson(2, n_normal)
})
 
# Fraudulent claims (manipulated values)
fraud_claims = pd.DataFrame({
    "claim_amount": np.random.normal(1500, 300, n_fraud),
    "num_services": np.random.poisson(10, n_fraud),
    "provider_rating": np.random.uniform(1.0, 2.5, n_fraud),
    "days_in_hospital": np.random.poisson(5, n_fraud)
})
 
df = pd.concat([normal_claims, fraud_claims], ignore_index=True)
 
# 2. Preprocess and scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
 
# 3. Train Isolation Forest (unsupervised anomaly detection)
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df["anomaly_score"] = iso_forest.fit_predict(X_scaled)
 
# 4. Mark anomalies (-1 = fraud)
df["fraud_detected"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)
 
# 5. Show summary
print("Detected Fraud Cases:\n")
print(df[df["fraud_detected"] == 1].head())
 
print(f"\nTotal Fraudulent Claims Detected: {df['fraud_detected'].sum()} out of {len(df)}")