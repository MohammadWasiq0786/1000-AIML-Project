"""
Project 817. Process Optimization Model

A process optimization model improves operational efficiency by adjusting process parameters to maximize output or minimize cost. In this example, we'll simulate a production process and use constrained optimization to find the best input combination for maximum output while respecting resource limits.

This model finds the optimal mix of inputs (x1 and x2) to maximize production, constrained by labor and material availability. In real cases, such models can optimize multi-stage processes, energy use, or even AI inference pipelines.
"""

from scipy.optimize import minimize
import numpy as np
 
# Objective: maximize production output
# Production function: output = 5*x1 + 8*x2 (x1, x2 are input units)
# Convert to minimization problem by negating the objective
def objective(x):
    return -1 * (5 * x[0] + 8 * x[1])  # maximize output
 
# Constraints:
# 1. Total labor available = 40 units
# 2. Total material available = 60 units
# Labor: x1 uses 1 unit, x2 uses 2 units
# Material: x1 uses 2 units, x2 uses 1 unit
constraints = [
    {'type': 'ineq', 'fun': lambda x: 40 - (1 * x[0] + 2 * x[1])},  # labor constraint
    {'type': 'ineq', 'fun': lambda x: 60 - (2 * x[0] + 1 * x[1])}   # material constraint
]
 
# Bounds: cannot use negative input units
bounds = [(0, None), (0, None)]  # x1, x2 ≥ 0
 
# Initial guess
x0 = [0, 0]
 
# Solve the optimization problem
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
 
# Display results
if result.success:
    x1, x2 = result.x
    max_output = -result.fun
    print(f"Optimal Inputs: x1 = {x1:.2f}, x2 = {x2:.2f}")
    print(f"Maximum Output: {max_output:.2f}")
else:
    print("Optimization failed:", result.message)