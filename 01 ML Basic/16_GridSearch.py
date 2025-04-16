"""
Project 16. Grid search for Hyperparameter Tuning
Description:
Grid Search is a brute-force approach to hyperparameter tuning, where we evaluate model performance over a predefined grid of parameters. This helps in finding the best combination that yields the highest accuracy. In this project, we use GridSearchCV to tune a Support Vector Machine (SVM) on the Iris dataset.

About:
🔍 What This Does:
Searches all combinations of C, gamma, and kernel

Uses 5-fold cross-validation to evaluate each combination

Selects the best hyperparameters and evaluates the model on the test set
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
 
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Define parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10],              # Regularization parameter
    'gamma': ['scale', 0.01, 0.001], # Kernel coefficient
    'kernel': ['linear', 'rbf']     # Kernel types
}
 
# Create an SVM model
model = SVC()
 
# Apply Grid Search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
 
# Best parameters and best score
print("Best Parameters Found:", grid_search.best_params_)
print(f"Best Cross-Validation Score: {grid_search.best_score_:.2f}")
 
# Evaluate on the test set using best estimator
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
 
# Final evaluation report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))