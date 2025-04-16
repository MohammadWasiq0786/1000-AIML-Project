"""
Project 10. Gradient boosting algorithm implementation
Description:
Gradient Boosting builds a strong learner by combining multiple weak learners (typically decision trees), each correcting the errors of the previous ones. It’s widely used in competitions and real-world applications due to its high accuracy and flexibility. In this project, we’ll use the Iris dataset and implement Gradient Boosting using scikit-learn.

About:
Gradient Boosting is:

Excellent for tabular data

Less prone to overfitting (with tuning)

Basis of powerful libraries like XGBoost, LightGBM, and CatBoost
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
 
# Load the Iris dataset
iris = load_iris()
X = iris.data        # Features
y = iris.target      # Labels: 0 = setosa, 1 = versicolor, 2 = virginica
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Initialize the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
 
# Train the model
gb_model.fit(X_train, y_train)
 
# Predict on the test data
y_pred = gb_model.predict(X_test)
 
# Evaluate the model
print("Gradient Boosting Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
 
# Predict a custom sample
sample = [[6.1, 2.8, 4.7, 1.2]]
prediction = gb_model.predict(sample)[0]
print(f"\nPrediction for sample {sample[0]}: {iris.target_names[prediction]}")
 
# Optional: Plot feature importance
import matplotlib.pyplot as plt
 
feature_importances = gb_model.feature_importances_
features = iris.feature_names
 
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.title("Gradient Boosting Feature Importance (Iris Dataset)")
plt.grid(True)
plt.show()