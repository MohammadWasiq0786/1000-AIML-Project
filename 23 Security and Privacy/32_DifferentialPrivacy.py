"""
Project 912. Differential Privacy Implementation

Differential Privacy (DP) provides a mathematical guarantee that the output of a computation doesn’t reveal too much about any individual in the dataset. In this project, we implement a basic differentially private mean computation by adding calibrated noise (Laplace or Gaussian) to the result.

What This Does:
Computes a private mean of the dataset using Laplace mechanism

Protects individuals by ensuring that the output doesn’t change much if any single entry is added/removed

📌 Key Concepts:

ε (epsilon): Lower means more privacy, but more noise

Sensitivity: Measures how much a single data point can change the output

Laplace noise: Added to "hide" individual contributions

🔐 Use cases:

Summaries of sensitive data (health, location, income)

DP in analytics dashboards and reports
"""

import numpy as np
 
# Simulated private data (e.g., income, test scores)
private_data = np.array([58, 72, 69, 85, 91, 60, 77, 95, 65, 88])
 
# True mean (not private)
true_mean = np.mean(private_data)
print(f"True Mean: {true_mean:.2f}")
 
# Differential Privacy settings
epsilon = 1.0  # privacy budget
sensitivity = (np.max(private_data) - np.min(private_data)) / len(private_data)
 
# Add Laplace noise
noise = np.random.laplace(loc=0, scale=sensitivity/epsilon)
dp_mean = true_mean + noise
 
print(f"DP Mean (ε = {epsilon}): {dp_mean:.2f}")