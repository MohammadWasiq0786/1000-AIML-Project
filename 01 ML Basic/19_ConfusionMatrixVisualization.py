"""
Project 19. Confusion matrix visualization
Description:
A confusion matrix gives a detailed breakdown of classification results—how many predictions were true positives, false positives, true negatives, and false negatives. This project implements a simple classification model and visualizes its confusion matrix using Seaborn heatmaps for better interpretability.

About:

🔍 What This Project Shows:
How the model performed in terms of:

True Positives (TP)

False Positives (FP)

True Negatives (TN)

False Negatives (FN)

Visualizes results using a heatmap for quick interpretation
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
 
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names
 
# Binary classification example: only setosa and versicolor (0 and 1)
X_binary = X[y != 2]
y_binary = y[y != 2]
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)
 
# Train logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Predict on test data
y_pred = model.predict(X_test)
 
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual Setosa", "Actual Versicolor"], columns=["Predicted Setosa", "Predicted Versicolor"])
 
# Print matrix
print("Confusion Matrix:\n")
print(cm_df)
 
# Visualize with seaborn heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Visualization")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()