"""
Project 24. Feature engineering toolkit
Description:
Feature engineering is the process of transforming raw data into meaningful features that improve model performance. In this project, we’ll build a toolkit that demonstrates key feature engineering techniques including polynomial features, interaction terms, binarization, discretization, and label encoding—all in one place using a synthetic dataset.

About:

🧰 What This Toolkit Includes:
PolynomialFeatures: Adds squared and interaction terms

Binarizer: Converts numerical values to 0/1

KBinsDiscretizer: Buckets continuous features

LabelEncoder: Converts categorical text to numeric values
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, Binarizer, KBinsDiscretizer, LabelEncoder
 
# Sample dataset
data = {
    'Age': [22, 25, 47, 52, 46],
    'Salary': [25000, 30000, 52000, 70000, 62000],
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes']  # Categorical label
}
df = pd.DataFrame(data)
print("Original Data:\n", df)
 
# ---- 1. Polynomial Features ----
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['Age', 'Salary']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Age', 'Salary']))
print("\nPolynomial Features (Degree 2):\n", poly_df)
 
# ---- 2. Binarization ----
binarizer = Binarizer(threshold=50000)
binarized_salary = binarizer.fit_transform(df[['Salary']])
df['High_Salary'] = binarized_salary
print("\nBinarized 'Salary' (1 if > 50,000):\n", df[['Salary', 'High_Salary']])
 
# ---- 3. Discretization (Bucketing Age) ----
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df['Age_Bin'] = discretizer.fit_transform(df[['Age']])
print("\nDiscretized 'Age' into 3 bins:\n", df[['Age', 'Age_Bin']])
 
# ---- 4. Label Encoding ----
label_encoder = LabelEncoder()
df['Purchased_Encoded'] = label_encoder.fit_transform(df['Purchased'])
print("\nLabel Encoded 'Purchased':\n", df[['Purchased', 'Purchased_Encoded']])