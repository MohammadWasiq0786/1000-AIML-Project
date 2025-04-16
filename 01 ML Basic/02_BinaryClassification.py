"""
Project 2. Logistic regression for binary classification
Description:
Logistic Regression is a supervised learning algorithm used for binary classification problems—where the output is either 0 or 1 (e.g., spam vs. ham, disease vs. no disease). It models the probability that a given input belongs to a particular class using a sigmoid function. In this project, we’ll use it to classify whether a student will pass (1) or fail (0) based on study hours.

About:
This basic logistic regression classifier shows how even a simple model can effectively separate binary outcomes. For real-world applications, you can use it for problems like disease prediction, customer churn, or email classification.
"""


# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
# Simulated data: Hours studied (X) vs. pass/fail (y)
# Label: 1 = Pass, 0 = Fail
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
 
# Create a logistic regression model
model = LogisticRegression()
 
# Train the model
model.fit(X, y)
 
# Predict probabilities for each input
y_prob = model.predict_proba(X)[:, 1]  # Probability of class "1" (Pass)
 
# Predict labels
y_pred = model.predict(X)
 
# Print the model parameters
print("Model Coefficient (Slope):", model.coef_[0][0])
print("Model Intercept:", model.intercept_[0])
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["Fail", "Pass"]))
 
# Predict outcome for a student who studied 5.5 hours
test_hours = [[5.5]]
predicted = model.predict(test_hours)[0]
probability = model.predict_proba(test_hours)[0][1]
 
print(f"\nStudent who studied 5.5 hours is predicted to: {'Pass' if predicted else 'Fail'}")
print(f"Probability of passing: {probability:.2f}")
 
# Plot the sigmoid curve
hours_range = np.linspace(0, 11, 100).reshape(-1, 1)
probabilities = model.predict_proba(hours_range)[:, 1]
 
plt.plot(hours_range, probabilities, color='blue', label='Probability of Passing')
plt.scatter(X, y, color='red', label='Training Data')
plt.axhline(0.5, color='gray', linestyle='--', label='Decision Boundary (0.5)')
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression - Pass/Fail Prediction")
plt.legend()
plt.grid(True)
plt.show()