"""
Project 840. A/B Testing Analysis System

An A/B testing analysis system evaluates the performance of two (or more) versions of a product, webpage, or feature to determine which performs better. In this project, we simulate user engagement data (e.g., conversion rates) and use statistical hypothesis testing to assess if version B is significantly better than version A.

Why This Works:
This model compares the conversion rates between A and B.

It uses a z-test to determine if the observed difference is due to chance.

The alternative='smaller' argument checks if B has a higher conversion rate than A.

You can expand this with:

Multi-variant tests

Bayesian A/B testing

Time-based segmentation (e.g., weekday/weekend)

Effect size and confidence intervals
"""

import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
 
# Simulated A/B test results
# A: control group, B: variant group
# conversions = number of users who performed desired action
conversions = [200, 240]     # A group had 200 conversions, B had 240
total_users = [1000, 1000]   # Each group had 1000 visitors
 
# Perform two-proportion z-test
z_stat, p_value = proportions_ztest(conversions, total_users, alternative='smaller')
 
print(f"Z-Statistic: {z_stat:.4f}")
print(f"P-Value: {p_value:.4f}")
 
# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant. Variant B outperforms A.")
else:
    print("Result: Not statistically significant. No clear winner.")