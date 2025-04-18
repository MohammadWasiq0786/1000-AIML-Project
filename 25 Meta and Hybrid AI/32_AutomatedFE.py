"""
Project 992: Automated Feature Engineering
Description
Feature engineering is the process of using domain knowledge to extract relevant features from raw data that improve model performance. Automated feature engineering automates this process, allowing us to discover and generate new features without manual intervention. In this project, we will use automated methods to generate meaningful features from raw data, which will be fed into a machine learning model.

Key Concepts Covered:
Automated Feature Engineering: Automatically creating new features or transforming existing features to improve the performance of machine learning models.

Imputation: Filling missing values in the dataset using strategies such as mean, median, or mode imputation.

Log Transformation: Applying a logarithmic transformation to variables that have a skewed distribution, improving model performance.

Power Transformation: Applying transformations (e.g., Box-Cox or Yeo-Johnson) to normalize features.

Discretization: Converting continuous variables into discrete bins or intervals to make the feature more informative.

Feature Scaling: Standardizing or normalizing features so they are on the same scale, preventing models from being biased toward larger-scale features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from feature_engine.transformation import LogTransformer, PowerTransformer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.scaling import StandardScaler
 
# Load a sample dataset (Boston housing dataset)
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='Target')
 
# Add some missing values to simulate a real-world scenario
X.iloc[5:10, 0] = np.nan
 
# Step 1: Impute missing values (mean imputation)
imputer = MeanMedianImputer(imputation_method='mean', variables=['CRIM'])
X_imputed = imputer.fit_transform(X)
 
# Step 2: Apply Log Transformation (for right-skewed features)
log_transformer = LogTransformer(variables=['CRIM', 'ZN', 'INDUS'])
X_log_transformed = log_transformer.fit_transform(X_imputed)
 
# Step 3: Apply Power Transformation (for normalization)
power_transformer = PowerTransformer(variables=['TAX', 'RAD'])
X_power_transformed = power_transformer.fit_transform(X_log_transformed)
 
# Step 4: Discretize a continuous feature (discretization into equal-frequency bins)
discretiser = EqualFrequencyDiscretiser(q=4, variables=['AGE'])
X_discretized = discretiser.fit_transform(X_power_transformed)
 
# Step 5: Feature Scaling (Standardize features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_discretized)
 
# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# Example output showing the effect of feature engineering on a few rows
print("Original Features (first 5 rows):")
print(X.head())
 
print("\nTransformed Features (first 5 rows):")
print(pd.DataFrame(X_scaled, columns=X.columns).head())