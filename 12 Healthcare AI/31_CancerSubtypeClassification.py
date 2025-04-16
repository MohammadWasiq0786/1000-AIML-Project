"""
Project 471. Cancer subtype classification
Description:
Cancer subtype classification helps identify specific molecular or histological types of cancer (e.g., HER2+, triple-negative breast cancer, etc.), which informs personalized treatment plans. In this project, we simulate a gene expression dataset and train a classifier to predict cancer subtypes.

About:
✅ What It Does:
Simulates gene expression data for 3 common breast cancer subtypes.

Uses a Random Forest classifier to predict subtype from expression profiles.

Can be extended to:

Use real omics data (RNA-seq, microarray)

Perform feature selection (e.g., DEGs, top 500 genes)

Visualize results with UMAP, t-SNE, or confusion matrices

In real-world use:

Use datasets like TCGA, METABRIC, or GEO

Apply methods like PCA, log normalization, or feature selection on omics data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# 1. Simulate gene expression data
np.random.seed(42)
n_samples = 300
n_genes = 50  # each gene is a feature
 
# Simulated gene expression matrix
X = np.random.rand(n_samples, n_genes)
 
# Simulated labels for 3 cancer subtypes
y = np.random.choice(["Luminal A", "HER2+", "Triple Negative"], size=n_samples)
 
# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 3. Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
 
# 4. Evaluate
y_pred = clf.predict(X_test)
print("Cancer Subtype Classification Report:\n")
print(classification_report(y_test, y_pred))