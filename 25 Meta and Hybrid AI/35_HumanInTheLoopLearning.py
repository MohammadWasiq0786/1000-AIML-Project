"""
Project 995: Human-in-the-loop Learning
Description
Human-in-the-loop (HITL) learning refers to the process of integrating human feedback into the machine learning model's training loop. This approach is particularly useful when automated systems make decisions that require human judgment or when the model needs to continuously improve with human input. In this project, we will create a human-in-the-loop system where a machine learning model actively learns from human-provided feedback to improve its predictions over time.

Key Concepts Covered:
Human-in-the-loop (HITL): In HITL learning, human feedback is used to correct or improve a machine learning model’s predictions, especially when the model is unsure or incorrect.

Active Learning: HITL is often paired with active learning, where the model queries for human feedback on uncertain predictions.

Continuous Improvement: The model continuously learns and adapts based on human corrections, improving over time.
"""


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
 
# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
 
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Initialize a simple Logistic Regression model
model = LogisticRegression(max_iter=1000)
 
# Train the model on the initial training data
model.fit(X_train, y_train)
 
# Evaluate the model's initial accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Accuracy: {accuracy:.4f}")
 
# Human-in-the-loop: Feedback Loop
# Let's simulate human feedback by allowing a user to correct the model's predictions.
# The model will be retrained with the corrected data (feedback).
# For simplicity, we'll randomly simulate whether the human agrees with the model's prediction.
 
# Function to simulate human feedback
def get_human_feedback(model, X_data, y_data):
    predictions = model.predict(X_data)
    feedback = []
    for i in range(len(predictions)):
        # Simulate feedback (e.g., random feedback for this example)
        human_agrees = random.choice([True, False])
        if not human_agrees:
            # If the human disagrees, assume they provide the correct label
            feedback.append(y_data.iloc[i])
        else:
            # If the human agrees, no correction is needed
            feedback.append(predictions[i])
    return feedback
 
# Simulate feedback on the model's predictions for the test set
feedback = get_human_feedback(model, X_test, y_test)
 
# Retrain the model using the feedback
model.fit(X_train.append(X_test), y_train.append(feedback))
 
# Reevaluate the model after incorporating human feedback
y_pred_new = model.predict(X_test)
accuracy_new = accuracy_score(y_test, y_pred_new)
print(f"Accuracy After Human Feedback: {accuracy_new:.4f}")