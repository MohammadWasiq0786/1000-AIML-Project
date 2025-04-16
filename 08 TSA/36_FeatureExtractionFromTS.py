"""
Project 316. Feature extraction from time series
Description:
Feature extraction transforms raw time series into a set of descriptive features (mean, variance, frequency domain, etc.) that can be used for:

Classification

Clustering

Anomaly detection

Forecasting (as regressors)

In this project, we’ll extract statistical and frequency-domain features using tsfresh, a powerful library for automatic time series feature engineering.

About:
✅ What It Does:
Simulates 50 time series samples from 2 classes (low vs. high frequency)

Extracts hundreds of statistical, autocorrelation, and frequency features

Converts time series into a feature matrix for ML models

Visualizes signal differences between the classes
"""

# Install if needed:
# pip install tsfresh
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
 
# 1. Simulate multiple time series samples (e.g., sensors or windows)
np.random.seed(42)
n_samples = 50
window_length = 100
 
# Generate time series data with labels
data = []
labels = []
for i in range(n_samples):
    freq = 1 if i < n_samples // 2 else 5
    signal = np.sin(np.linspace(0, 2 * np.pi * freq, window_length)) + 0.1 * np.random.randn(window_length)
    for t in range(window_length):
        data.append({'id': i, 'time': t, 'value': signal[t]})
    labels.append(0 if i < n_samples // 2 else 1)  # binary classes
 
df = pd.DataFrame(data)
 
# 2. Extract time series features
features = extract_features(df, column_id='id', column_sort='time')
features = impute(features)  # handle NaNs
 
# 3. Inspect some features
print("✅ Extracted Features:")
print(features.head())
 
# 4. Plot one sample from each class
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(df[df['id'] == 0]['value'])
plt.title("Class 0 Sample (Low Frequency)")
 
plt.subplot(1, 2, 2)
plt.plot(df[df['id'] == 30]['value'])
plt.title("Class 1 Sample (High Frequency)")
plt.tight_layout()
plt.show()