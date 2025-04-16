"""
Project 18. Correlation analysis tool
Description:
Correlation analysis is used to measure and visualize the strength and direction of relationships between numerical variables. This project builds a simple tool to compute and visualize correlations using Pearson correlation coefficients and a heatmap. It helps in feature selection and understanding data relationships.

About:
✅ What This Tool Does:
Calculates pairwise correlations between features

Prints correlation matrix

Visualizes relationships using a color-coded heatmap
"""


# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
# Sample dataset (can also be replaced with any CSV)
data = {
    "Age": [25, 32, 47, 51, 62, 23, 43, 36, 52, 48],
    "Income": [45000, 54000, 72000, 80000, 88000, 43000, 69000, 62000, 85000, 79000],
    "SpendingScore": [30, 40, 50, 60, 65, 25, 45, 42, 58, 62],
    "CreditScore": [650, 680, 720, 700, 710, 640, 690, 685, 705, 695]
}
 
# Convert to DataFrame
df = pd.DataFrame(data)
 
# Calculate the correlation matrix using Pearson method
correlation_matrix = df.corr()
 
# Print correlation matrix
print("Correlation Matrix:\n")
print(correlation_matrix)
 
# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
