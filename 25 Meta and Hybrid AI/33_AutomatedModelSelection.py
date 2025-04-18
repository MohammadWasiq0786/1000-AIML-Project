"""
Project 993: Automated Model Selection
Description
Automated model selection is a process where the best model for a given dataset is selected automatically based on the dataset's characteristics. This process evaluates multiple machine learning algorithms, tunes their hyperparameters, and selects the best-performing model. In this project, we will use TPOT (Tree-based Pipeline Optimization Tool), an automated machine learning (AutoML) library that performs hyperparameter tuning and model selection efficiently.

Key Concepts Covered:
Automated Model Selection: Using libraries like TPOT, we automatically search through various models and their hyperparameters to select the one that performs best for the given data.

Hyperparameter Tuning: In addition to selecting the best model, TPOT also optimizes hyperparameters using genetic algorithms to improve performance.

AutoML: Automated Machine Learning (AutoML) frameworks like TPOT allow non-experts to quickly build effective machine learning models without manually tuning hyperparameters or selecting algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score
 
# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Initialize the TPOTClassifier with a generation of 5 and population size of 20
# This means it will create 5 generations with 20 models in each generation
tpot = TPOTClassifier(Generations=5, Population_size=20, random_state=42, verbosity=2)
 
# Fit the TPOTClassifier to the training data (automated model selection)
tpot.fit(X_train, y_train)
 
# Evaluate the model on the test data
y_pred = tpot.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the best model found by TPOT: {accuracy:.4f}")
 
# Export the best model pipeline found by TPOT
tpot.export('best_model_pipeline.py')