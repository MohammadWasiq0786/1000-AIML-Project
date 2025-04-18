"""
Project 984: Causal Inference in Machine Learning
Description
Causal inference is the process of determining whether a relationship between two variables is causal rather than just correlational. In this project, we will implement a causal inference model to identify causal relationships in data, using techniques such as propensity score matching and causal graphs.

Key Concepts Covered:
Propensity Score Matching (PSM): A technique used in observational studies to match treated and untreated units based on similar propensity scores to estimate causal effects.

Causal Inference: The process of identifying whether a relationship between two variables is causal (i.e., treatment causes the outcome) or merely correlational.

Causal Effects: Estimating the effect of a treatment or intervention, typically by comparing outcomes of treated vs. untreated units after controlling for confounding variables.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Generate synthetic data for causal inference
np.random.seed(0)
n = 1000  # Number of samples
X = np.random.randn(n, 3)  # 3 features
 
# Create treatment variable (binary)
T = np.random.binomial(1, p=0.5, size=n)
 
# Define a true causal effect
Y0 = X[:, 0] + X[:, 1]  # Outcome without treatment
Y1 = Y0 + 2 * X[:, 2]   # Outcome with treatment (treatment effect on X2)
 
# Outcome variable (Y) is observed depending on treatment
Y = Y0 + T * (Y1 - Y0)  # Observed outcome
 
# Convert to DataFrame for convenience
df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
df["T"] = T
df["Y"] = Y
 
# Step 1: Estimate the propensity score using logistic regression
X = sm.add_constant(df[["X1", "X2", "X3"]])  # Add constant for intercept
logit = LogisticRegression()
logit.fit(df[["X1", "X2", "X3"]], df["T"])
 
# Predicted propensity scores
df["propensity_score"] = logit.predict_proba(df[["X1", "X2", "X3"]])[:, 1]
 
# Step 2: Perform propensity score matching
treated = df[df["T"] == 1]
untreated = df[df["T"] == 0]
 
# Sort by propensity score
treated = treated.sort_values("propensity_score")
untreated = untreated.sort_values("propensity_score")
 
# Matching treated and untreated units based on the closest propensity scores
matches = []
for _, t_row in treated.iterrows():
    closest_match = untreated.iloc[(untreated["propensity_score"] - t_row["propensity_score"]).abs().argmin()]
    matches.append((t_row, closest_match))
 
# Step 3: Estimate the causal effect
treated_outcomes = [match[0]["Y"] for match in matches]
control_outcomes = [match[1]["Y"] for match in matches]
 
# Calculate the average treatment effect (ATE)
ATE = np.mean(np.array(treated_outcomes) - np.array(control_outcomes))
print(f"Estimated Average Treatment Effect (ATE): {ATE:.4f}")
 
# Step 4: Plot the distribution of propensity scores for treated vs untreated
plt.figure(figsize=(8, 6))
plt.hist(treated["propensity_score"], bins=30, alpha=0.5, label="Treated", color='blue')
plt.hist(untreated["propensity_score"], bins=30, alpha=0.5, label="Untreated", color='red')
plt.xlabel("Propensity Score")
plt.ylabel("Frequency")
plt.legend()
plt.title("Distribution of Propensity Scores")
plt.show()