"""
Project 813. Workforce Planning Model

A workforce planning model helps determine the number of employees needed across departments or future time periods based on projected demand. In this example, we simulate staffing demand over upcoming quarters and optimize hiring plans to meet demand at minimum cost using linear programming.

This basic model calculates how many people to hire each quarter to meet demand at the lowest cost. In advanced scenarios, you can incorporate attrition, part-time/full-time mixes, training lags, and skill-based assignments.
"""

import numpy as np
from scipy.optimize import linprog
import pandas as pd
 
# Assume we have to plan staffing over 3 quarters
# Forecasted demand for employees in each quarter
demand = [50, 60, 55]  # Q1, Q2, Q3
 
# Hiring cost per employee per quarter
hiring_cost = [1000, 1100, 1200]
 
# Objective: minimize hiring costs while meeting demand
# Decision variables: number of employees to hire each quarter
c = hiring_cost
 
# Each quarter must meet or exceed cumulative demand
# Construct inequality matrix to reflect cumulative hiring
A = [
    [-1, 0, 0],       # Q1 hires >= 50
    [-1, -1, 0],      # Q1 + Q2 hires >= 60
    [-1, -1, -1]      # Q1 + Q2 + Q3 hires >= 55
]
b = [-d for d in demand]  # Flip sign to use <= format in linprog
 
# Bounds: cannot hire negative number of employees
bounds = [(0, None), (0, None), (0, None)]
 
# Solve the linear programming problem
result = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
 
# Output the hiring plan
if result.success:
    hires = np.ceil(result.x)  # Round up since we can't hire fractions of people
    df = pd.DataFrame({
        'Quarter': ['Q1', 'Q2', 'Q3'],
        'Hires': hires.astype(int),
        'Cost per Hire': c,
        'Total Cost': hires * c
    })
    print("Optimal Workforce Hiring Plan:")
    print(df)
    print(f"\nTotal Hiring Cost: ${df['Total Cost'].sum():.2f}")
else:
    print("Optimization failed:", result.message)