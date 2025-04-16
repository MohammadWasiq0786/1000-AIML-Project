"""
Project 60. A/B testing analyzer
Description:
An A/B testing analyzer helps evaluate whether two variants (A and B) have significantly different performance, such as conversion rates. In this project, we build a tool that compares outcomes using a two-proportion z-test to determine if the difference between versions is statistically significant.

About:
🧪 What This Project Demonstrates:
Compares two conversion rates using z-test for proportions

Calculates and interprets p-values and statistical significance

Visualizes results for quick decision-making
"""

from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
 
# Sample A/B test data
# Variant A: control group
conversions_A = 120
visitors_A = 2400
 
# Variant B: test group
conversions_B = 150
visitors_B = 2300
 
# Combine data
counts = [conversions_A, conversions_B]  # successes
nobs = [visitors_A, visitors_B]          # total observations
 
# Run two-proportion z-test
z_stat, p_value = proportions_ztest(count=counts, nobs=nobs)
 
# Display results
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
 
alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant difference between A and B (Reject Null Hypothesis)")
else:
    print("Result: No statistically significant difference (Fail to Reject Null Hypothesis)")
 
# Optional: Visualize conversion rates
labels = ['Variant A', 'Variant B']
conversion_rates = [conversions_A / visitors_A, conversions_B / visitors_B]
 
plt.bar(labels, conversion_rates, color=['blue', 'green'])
plt.title('Conversion Rate Comparison: A/B Test')
plt.ylabel('Conversion Rate')
plt.ylim(0, max(conversion_rates) + 0.02)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()