"""
Project 8. Support vector machine for classification
Description:
Support Vector Machines (SVM) are powerful classification algorithms that find the optimal hyperplane that separates different classes in feature space. They are particularly effective for high-dimensional data and complex decision boundaries. In this project, we’ll use SVM to classify data from the Iris dataset.

About:
SVM works well for both linear and non-linear decision boundaries. You can:

Experiment with kernel types (linear, RBF, polynomial).

Adjust C (regularization) and gamma (kernel coefficient) for tuning.

Try visualizing 2D SVM boundaries using a subset of features.
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
 
# Load the Iris dataset
iris = load_iris()
X = iris.data        # Features
y = iris.target      # Labels: 0 = setosa, 1 = versicolor, 2 = virginica
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Create an SVM classifier with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Try 'linear', 'poly', 'sigmoid', or 'rbf'
 
# Train the model
svm_model.fit(X_train, y_train)
 
# Predict the test set
y_pred = svm_model.predict(X_test)
 
# Evaluate the performance
print("SVM Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
 
# Predict a new sample
sample = [[6.0, 2.9, 4.5, 1.5]]
prediction = svm_model.predict(sample)[0]
print(f"\nPrediction for sample {sample[0]}: {iris.target_names[prediction]}")