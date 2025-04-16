"""
Project 9. Random forest classifier
Description:
A Random Forest is an ensemble learning method that builds multiple decision trees and merges their predictions for improved accuracy and robustness. It reduces overfitting and handles both categorical and numerical data effectively. In this project, we’ll use the Iris dataset to train and evaluate a Random Forest classifier.

About:
Random Forests are fast, accurate, and easy to tune, and they also:

Provide feature importance scores

Handle missing data well

Work great for imbalanced datasets (with tweaks)
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
 
# Load the Iris dataset
iris = load_iris()
X = iris.data        # Features
y = iris.target      # Target labels
 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Create a Random Forest classifier with 100 trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
 
# Train the model
rf_model.fit(X_train, y_train)
 
# Predict on the test set
y_pred = rf_model.predict(X_test)
 
# Evaluate the model
print("Random Forest Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
 
# Predict a custom sample
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = rf_model.predict(sample)[0]
print(f"\nPrediction for sample {sample[0]}: {iris.target_names[prediction]}")
 
# Optional: Display feature importance
import matplotlib.pyplot as plt
 
feature_importances = rf_model.feature_importances_
features = iris.feature_names
 
# Plot the feature importances
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Iris Dataset)")
plt.grid(True)
plt.show()