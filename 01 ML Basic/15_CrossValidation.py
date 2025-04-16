"""
Project 15. Cross-validation implementation
Description:
Cross-validation is a powerful model evaluation technique used to assess a model’s generalization capability. Instead of relying on a single train-test split, it partitions the data into multiple folds and evaluates performance across them. In this project, we demonstrate k-fold cross-validation using the Iris dataset and a logistic regression model.

About:
✅ What This Code Does:
Loads the Iris dataset.

Trains and evaluates a Logistic Regression model using 5-fold cross-validation.

Computes and prints the accuracy for each fold, along with the mean and standard deviation.

Variations You Can Try:
Use different scoring metrics like f1, precision, recall.

Try StratifiedKFold to preserve label distribution.

Use cross_val_predict to get predicted labels for analysis.
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
 
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Create a Logistic Regression model
model = LogisticRegression(max_iter=200)
 
# Define 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
 
# Perform cross-validation and get accuracy for each fold
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
 
# Output results
print("Cross-Validation Accuracies (5-Fold):", cv_scores)
print(f"Mean Accuracy: {np.mean(cv_scores):.2f}")
print(f"Standard Deviation: {np.std(cv_scores):.2f}")