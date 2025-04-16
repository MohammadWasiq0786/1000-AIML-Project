"""
Project 4. Decision tree classifier for Iris dataset
Description:
A Decision Tree Classifier uses a tree-like model to make decisions based on feature values. In this project, we’ll use the famous Iris dataset to classify flower species (Setosa, Versicolor, Virginica) based on features like petal and sepal length/width. This model is easy to interpret and visualize.

About:
This decision tree classifier is perfect for understanding how machine learning makes decisions based on feature thresholds. It's also easily visualizable, making it ideal for teaching, model interpretability, or quick prototyping.
"""

# Import required libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
 
# Load the Iris dataset
iris = load_iris()
X = iris.data      # Features: sepal length, sepal width, petal length, petal width
y = iris.target    # Labels: 0 = setosa, 1 = versicolor, 2 = virginica
 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Create and train the decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
 
# Predict on the test set
y_pred = model.predict(X_test)
 
# Evaluate the model
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
 
# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show()