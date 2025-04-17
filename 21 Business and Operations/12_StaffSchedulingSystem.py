"""
Project 812. Staff Scheduling System

Staff scheduling involves assigning employees to shifts or tasks to meet operational requirements while minimizing overwork or understaffing. In this simplified example, we'll use linear programming to assign staff to shifts such that all shifts are covered with minimal total working hours.

This model ensures every shift is covered once while assigning available employees. For real-world use, integer programming and constraints for max hours, rest periods, or employee preferences would be added.
"""

import numpy as np
from scipy.optimize import linprog
import pandas as pd
 
# Assume we have 3 employees and 4 shifts to fill
# Cost matrix (e.g., hours employee is available to work each shift)
# Rows = employees (E1, E2, E3), Columns = shifts (S1, S2, S3, S4)
availability_cost = [
    [1, 1, 0, 1],   # E1 can work S1, S2, S4
    [1, 0, 1, 1],   # E2 can work S1, S3, S4
    [0, 1, 1, 1]    # E3 can work S2, S3, S4
]
 
# Flatten the cost matrix (objective: minimize total shifts assigned)
c = np.array(availability_cost).flatten()
 
# Constraints: each shift must be assigned exactly once
A_eq = []
b_eq = []
 
# For each shift, sum over employees = 1 (exactly one employee per shift)
for j in range(4):  # 4 shifts
    row = [0] * 12  # 3x4 = 12 variables
    for i in range(3):  # 3 employees
        row[i * 4 + j] = 1
    A_eq.append(row)
    b_eq.append(1)
 
# Bounds: either an employee works a shift (1) or not (0)
bounds = [(0, 1)] * 12
 
# Solve the integer linear program (relaxed to continuous 0–1 for simplicity)
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
 
# Output the schedule
if result.success:
    x = result.x.round()  # round for binary interpretation
    schedule = np.reshape(x, (3, 4))
    df = pd.DataFrame(schedule, 
                      index=['Employee 1', 'Employee 2', 'Employee 3'], 
                      columns=['Shift 1', 'Shift 2', 'Shift 3', 'Shift 4'])
    print("Optimal Staff Scheduling (1 = assigned):")
    print(df.astype(int))
else:
    print("Scheduling failed:", result.message)