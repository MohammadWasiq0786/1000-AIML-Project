"""
Project 815. Project Risk Assessment

Project risk assessment evaluates potential risks that could affect a project's success — such as delays, budget overruns, or resource shortages. A basic model quantifies risk by evaluating the probability and impact of various risk factors and calculating a risk score to prioritize them.

This model helps identify which risks should be monitored or mitigated first, based on a simple risk scoring system. You can expand it by adding risk mitigation strategies, status tracking, or dynamic updates during project execution.
"""

import pandas as pd
 
# Define a list of risks with estimated probability and impact (both on a 1–5 scale)
data = {
    'Risk': [
        'Resource Shortage',
        'Budget Overrun',
        'Technology Failure',
        'Vendor Delay',
        'Scope Creep',
        'Regulatory Issue'
    ],
    'Probability': [4, 3, 2, 3, 5, 1],  # Likelihood of occurrence
    'Impact': [5, 4, 3, 4, 2, 5]        # Severity if it occurs
}
 
df = pd.DataFrame(data)
 
# Calculate risk score as the product of probability and impact
df['Risk Score'] = df['Probability'] * df['Impact']
 
# Sort risks by highest risk score
df_sorted = df.sort_values(by='Risk Score', ascending=False)
 
# Display risk assessment table
print("Project Risk Assessment Table (Sorted by Risk Score):")
print(df_sorted)
 
# Visual representation (bar chart)
import matplotlib.pyplot as plt
 
plt.figure(figsize=(10, 6))
plt.barh(df_sorted['Risk'], df_sorted['Risk Score'], color='salmon')
plt.xlabel('Risk Score (Probability × Impact)')
plt.title('Project Risk Prioritization')
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()