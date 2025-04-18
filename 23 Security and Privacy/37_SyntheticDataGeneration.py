"""
Project 917. Synthetic Data Generation

Synthetic data generation creates artificial datasets that resemble real data but do not contain any actual personal records—useful for privacy-preserving analysis, ML model training, or testing.

In this project, we generate synthetic tabular data using scikit-learn’s make_classification, which mimics a real-world binary classification dataset.

What It Demonstrates:
Generates structured data that preserves statistical properties of real data.

No original records used—preserves privacy.

Useful for testing, ML training, and data sharing.

🧠 Advanced Tools:

CTGAN / TVAE via SDV library (https://sdv.dev/)

Gretel.ai, Hazy, or Mostly AI for enterprise-grade synthetic data

Use in healthcare, finance, or regulated industries
"""

import pandas as pd
from sklearn.datasets import make_classification
 
# Generate synthetic classification data
X, y = make_classification(
    n_samples=100,       # number of synthetic records
    n_features=5,        # number of features
    n_informative=3,     # informative features
    n_redundant=1,       # redundant features
    n_classes=2,         # binary classification
    random_state=42
)
 
# Convert to DataFrame
feature_names = ['Income', 'Age', 'DebtRatio', 'CreditScore', 'LoanAmount']
df = pd.DataFrame(X, columns=feature_names)
df['Default'] = y  # target variable
 
# Show synthetic data
print("🧪 Synthetic Data Sample:")
print(df.head())