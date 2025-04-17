"""
Project 516: Financial Planning Assistant
Description:
A financial planning assistant helps individuals plan and manage their finances by offering budgeting, savings goals, and investment advice. In this project, we will create a simple financial planning assistant that helps users manage their monthly expenses, set savings goals, and track their progress.
"""

import pandas as pd
import numpy as np
 
# 1. Simulate monthly expenses data
np.random.seed(42)
categories = ['Rent', 'Groceries', 'Utilities', 'Entertainment', 'Transportation', 'Insurance', 'Miscellaneous']
expenses = np.random.randint(100, 1500, size=7)
 
df = pd.DataFrame({'Category': categories, 'Expense': expenses})
 
# 2. Calculate total expenses and set savings goal
total_expenses = df['Expense'].sum()
monthly_income = 4000  # Example monthly income
savings_goal = 0.2 * monthly_income  # 20% of income for savings
 
# 3. Calculate remaining income after expenses
remaining_income = monthly_income - total_expenses
 
# 4. Financial advice based on savings goal and remaining income
if remaining_income >= savings_goal:
    advice = "You are on track with your savings goal!"
else:
    advice = "Consider reducing some expenses to meet your savings goal."
 
# 5. Display the financial summary and advice
print("Monthly Expenses Breakdown:")
print(df)
print(f"\nTotal Monthly Expenses: ${total_expenses}")
print(f"Monthly Income: ${monthly_income}")
print(f"Savings Goal (20% of income): ${savings_goal}")
print(f"Remaining Income: ${remaining_income}")
print(f"\nFinancial Advice: {advice}")