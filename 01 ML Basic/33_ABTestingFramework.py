"""
Project 33. A/B testing framework
Description:
A/B testing is a method of comparing two versions (A and B) of a variable to determine which one performs better. It's commonly used in product design, marketing, and UX optimization. In this project, we create a Python-based framework to simulate and analyze an A/B test using statistical hypothesis testing (e.g., t-test or z-test for proportions).

About:
🧪 What This Project Teaches:
Simulates conversion data

Applies both T-Test and Z-Test for proportions

Calculates and interprets p-values, conversion lift, and significance

Visualizes the results
"""

import numpy as np
from scipy.stats import ttest_ind, norm
import matplotlib.pyplot as plt
 
# Simulate user conversion data for two versions of a web page
np.random.seed(42)
 
# Group A (Control): 1000 users, 12% conversion rate
group_a = np.random.binomial(1, 0.12, 1000)
 
# Group B (Variant): 1000 users, 15% conversion rate
group_b = np.random.binomial(1, 0.15, 1000)
 
# Calculate basic statistics
conv_rate_a = np.mean(group_a)
conv_rate_b = np.mean(group_b)
 
print(f"Conversion Rate A (Control): {conv_rate_a:.3f}")
print(f"Conversion Rate B (Variant): {conv_rate_b:.3f}")
print(f"Lift: {(conv_rate_b - conv_rate_a) / conv_rate_a * 100:.2f}%")
 
# ---- Option 1: T-Test (for normally distributed metrics like revenue/time) ----
t_stat, p_value_t = ttest_ind(group_a, group_b)
print(f"\nT-Test p-value: {p_value_t:.4f}")
 
# ---- Option 2: Z-Test for Proportions ----
n_a, n_b = len(group_a), len(group_b)
success_a, success_b = sum(group_a), sum(group_b)
 
# Pooled probability
p_pool = (success_a + success_b) / (n_a + n_b)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
z = (conv_rate_b - conv_rate_a) / se
p_value_z = 1 - norm.cdf(z)
 
print(f"Z-Test z-score: {z:.2f}")
print(f"Z-Test one-tailed p-value: {p_value_z:.4f}")
 
# Plot distributions
plt.figure(figsize=(8, 4))
plt.bar(["Control (A)", "Variant (B)"], [conv_rate_a, conv_rate_b], color=["blue", "green"])
plt.title("A/B Test Conversion Rates")
plt.ylabel("Conversion Rate")
plt.grid(axis='y')
plt.tight_layout()
plt.show()