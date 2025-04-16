"""
Project 23. Data normalization techniques
Description:
Data normalization transforms features to a common scale without distorting differences in the ranges of values. It is critical for models that are sensitive to scale (e.g., KNN, SVM, Neural Networks). In this project, we’ll demonstrate and compare Min-Max Scaling, Z-score Standardization, and Max Abs Scaling using a synthetic dataset.

About:

🧠 What This Teaches:
Min-Max Scaling: Rescales values to [0, 1]

Z-Score Standardization: Centers around mean (0) and scales to unit variance

Max-Abs Scaling: Scales by maximum absolute value (retains sparsity)
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
 
# Create a sample dataset with varying scales
data = {
    'Feature_1': [100, 200, 300, 400, 500],   # Large range
    'Feature_2': [1, 2, 3, 4, 5],             # Small range
    'Feature_3': [-50, -25, 0, 25, 50]        # Negative to positive
}
 
df = pd.DataFrame(data)
print("Original Data:\n", df)
 
# ---- Min-Max Scaling ----
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)
print("\nMin-Max Scaled Data:\n", df_minmax)
 
# ---- Z-score Standardization ----
standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)
print("\nZ-Score Standardized Data:\n", df_standard)
 
# ---- Max-Abs Scaling ----
maxabs_scaler = MaxAbsScaler()
df_maxabs = pd.DataFrame(maxabs_scaler.fit_transform(df), columns=df.columns)
print("\nMax-Abs Scaled Data:\n", df_maxabs)