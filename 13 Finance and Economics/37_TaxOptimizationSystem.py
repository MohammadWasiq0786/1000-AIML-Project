"""
Project 517: Tax Optimization System
Description:
A tax optimization system helps individuals or businesses reduce their tax liabilities by suggesting optimal strategies based on income, deductions, and tax rates. In this project, we will create a simple system that suggests tax-saving methods like deductions for expenses, investments in tax-efficient instruments, and tax credits.
"""

import pandas as pd
import numpy as np
 
# 1. Simulate income and tax data
np.random.seed(42)
income = np.random.randint(50000, 150000, 1000)  # Simulate annual income
tax_rate = 0.2  # Assume a flat 20% tax rate for simplicity
 
# 2. Simulate deductions and tax credits (e.g., education, health expenses, investments)
deductions = np.random.randint(1000, 5000, 1000)  # Simulate tax-deductible expenses
tax_credits = np.random.randint(200, 1000, 1000)  # Simulate tax credits (e.g., for children, education)
 
# 3. Calculate tax before and after optimization
tax_before_optimization = income * tax_rate
tax_after_optimization = (income - deductions) * tax_rate - tax_credits
 
# 4. Create a DataFrame to display the results
df = pd.DataFrame({
    'Income': income,
    'Deductions': deductions,
    'Tax Credits': tax_credits,
    'Tax Before Optimization': tax_before_optimization,
    'Tax After Optimization': tax_after_optimization
})
 
# 5. Show the tax optimization results
df['Tax Savings'] = df['Tax Before Optimization'] - df['Tax After Optimization']
 
# Display the summary of tax savings
print("Tax Optimization Summary (Top 5 entries):")
print(df[['Income', 'Deductions', 'Tax Credits', 'Tax Savings']].head())
 
# 6. Calculate the average tax savings across the dataset
average_tax_savings = df['Tax Savings'].mean()
print(f"\nAverage Tax Savings per Individual: ${average_tax_savings:.2f}")