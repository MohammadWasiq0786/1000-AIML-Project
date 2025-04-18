"""
Project 996: Interactive Machine Learning
Description
Interactive machine learning (IML) allows users to actively participate in the learning process. It is designed to make the machine learning process more collaborative, where the system can ask for feedback or input from a user, and use that input to refine the model in real-time. This project will demonstrate a simple interactive machine learning system where the user provides feedback on the model’s predictions, and the model learns and improves over time.

Key Concepts Covered:
Interactive Machine Learning (IML): IML allows users to provide feedback on model predictions in real-time, enhancing the learning process.

Active Learning: The model queries the user to correct predictions, improving its performance over time.

User-Centric Learning: The model continuously adapts based on the user’s input, creating a dynamic learning environment.
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
 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Initialize a Logistic Regression model
model = LogisticRegression(max_iter=1000)
 
# Train the model on the initial training data
model.fit(X_train, y_train)
 
# Evaluate the model's accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Accuracy: {accuracy:.4f}")
 
# Function to simulate interactive feedback from the user
def get_user_feedback(X_data, y_data, model):
    """
    Simulate user feedback where the user might agree or correct the model’s prediction.
    For simplicity, feedback is simulated by random chance.
    If the model's prediction is wrong, the user provides the correct label.
    """
    feedback = []
    for i in range(len(X_data)):
        prediction = model.predict([X_data.iloc[i]])[0]
        true_label = y_data.iloc[i]
        
        # Simulate user feedback
        if prediction != true_label:
            feedback.append(true_label)  # If wrong, the user provides the correct label
        else:
            feedback.append(prediction)  # If correct, no change needed
    
    return feedback
 
# Simulate user feedback on the model's predictions for the test set
feedback = get_user_feedback(X_test, y_test, model)
 
# Retrain the model using the feedback from the user
model.fit(X_train.append(X_test), y_train.append(feedback))
 
# Evaluate the model after incorporating user feedback
y_pred_new = model.predict(X_test)
accuracy_new = accuracy_score(y_test, y_pred_new)
print(f"Accuracy After User Feedback: {accuracy_new:.4f}")