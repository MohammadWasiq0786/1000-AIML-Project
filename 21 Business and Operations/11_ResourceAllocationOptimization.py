"""
Project 811. Resource Allocation Optimization

This project aims to optimally allocate limited resources (e.g., budget, labor, machines) to various tasks or departments to maximize output or minimize cost. We'll simulate a scenario where resources need to be allocated across projects for maximum profit using linear programming.

This linear program allocates limited labor hours to maximize profit while respecting both total and per-project constraints. You can expand this to include multiple resources, skill levels, or multi-objective goals using more advanced solvers.
"""

from scipy.optimize import linprog
import numpy as np
import pandas as pd
 
# Example: Allocate labor hours across 3 projects to maximize profit
# Profit per unit of labor hour for each project
profits = [-50, -40, -70]  # Negative for maximization in linprog
 
# Constraints:
# Total available labor = 100 hours
# Each project has max hours it can use
A = [
    [1, 1, 1],       # Total labor constraint
    [1, 0, 0],       # Project A max
    [0, 1, 0],       # Project B max
    [0, 0, 1]        # Project C max
]
 
b = [100, 40, 50, 30]  # total labor, max per project
 
# Bounds for each variable (hours assigned to each project)
bounds = [(0, None), (0, None), (0, None)]
 
# Solve the linear programming problem
result = linprog(c=profits, A_ub=A, b_ub=b, bounds=bounds, method='highs')
 
# Output results
if result.success:
    hours = result.x
    total_profit = -result.fun  # flip sign back to positive
    print("Optimal Resource Allocation (Labor Hours):")
    df = pd.DataFrame({
        'Project': ['A', 'B', 'C'],
        'Allocated Hours': hours.round(2),
        'Profit/Hour': [-p for p in profits],
        'Total Profit': (hours * [-p for p in profits]).round(2)
    })
    print(df)
    print(f"\nTotal Maximum Profit: ${total_profit:.2f}")
else:
    print("Optimization failed:", result.message)