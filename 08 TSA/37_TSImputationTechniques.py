"""
Project 317. Time series imputation techniques
Description:
Time series imputation is the process of filling in missing values in temporal data. This is critical because missing values can:

Break models

Distort trends

Mislead anomaly detection

In this project, we’ll explore common imputation strategies including:

Forward fill

Interpolation

Model-based (KNN)

About:
✅ What It Does:
Simulates a noisy sine wave with random missing values

Applies 3 imputation techniques:

Forward fill (carry last known value)

Linear interpolation (estimate between known points)

KNN imputation (learn from neighbors)

Compares results visually

This is essential before applying forecasting, classification, or clustering on time series data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
 
# 1. Generate sine wave and introduce missing values
np.random.seed(42)
t = np.linspace(0, 10, 200)
data = np.sin(t) + 0.1 * np.random.randn(len(t))
 
# Introduce missing values (20%)
missing_idx = np.random.choice(len(t), size=int(0.2 * len(t)), replace=False)
data_missing = data.copy()
data_missing[missing_idx] = np.nan
 
# 2. Imputation methods
df = pd.DataFrame({'value': data_missing})
 
# a) Forward fill
ffill = df.fillna(method='ffill')
 
# b) Interpolation
interp = df.interpolate(method='linear')
 
# c) KNN Imputer
knn_imputer = KNNImputer(n_neighbors=3)
knn = knn_imputer.fit_transform(df)
 
# 3. Plot all imputations
plt.figure(figsize=(12, 5))
plt.plot(data, label='Original', alpha=0.5)
plt.plot(df.index, df['value'], 'o', label='Missing', color='gray', alpha=0.3)
plt.plot(ffill, label='Forward Fill')
plt.plot(interp, label='Linear Interpolation')
plt.plot(knn, label='KNN Imputer')
plt.title("Time Series Imputation Methods")
plt.legend()
plt.grid(True)
plt.show()