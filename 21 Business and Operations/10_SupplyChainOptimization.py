"""
Project 810. Supply Chain Optimization

Supply chain optimization focuses on minimizing cost and maximizing efficiency across sourcing, production, storage, and delivery. In this simplified project, we simulate a transportation problem — deciding how to ship products from warehouses to stores at the lowest cost using linear programming.

This code determines the most cost-effective distribution plan to satisfy demand while respecting warehouse supply constraints. For more complex networks, tools like PuLP or Google OR-Tools can handle multi-stage supply chains, constraints, and objectives.
"""

import numpy as np
from scipy.optimize import linprog
import pandas as pd
 
# Supply (units available at each warehouse)
supply = [100, 150]  # Warehouse A, Warehouse B
 
# Demand (units needed at each store)
demand = [80, 120, 50]  # Store X, Store Y, Store Z
 
# Cost matrix (shipping cost per unit from each warehouse to each store)
cost_matrix = [
    [2, 4, 5],   # Warehouse A to X, Y, Z
    [3, 1, 7]    # Warehouse B to X, Y, Z
]
 
# Flatten cost matrix for optimization
c = np.array(cost_matrix).flatten()
 
# Define inequality constraints (supply limits per warehouse)
A_eq = []
 
# Supply constraints: rows = warehouses, columns = flattened shipping plan
for i in range(len(supply)):
    row = [0] * len(c)
    for j in range(len(demand)):
        row[i * len(demand) + j] = 1
    A_eq.append(row)
 
# Demand constraints: columns = stores
for j in range(len(demand)):
    row = [0] * len(c)
    for i in range(len(supply)):
        row[i * len(demand) + j] = 1
    A_eq.append(row)
 
A_eq = np.array(A_eq)
b_eq = np.array(supply + demand)
 
# Solve the linear programming problem
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
 
# Output results
if result.success:
    print("Optimal Shipping Plan (units from warehouse to store):")
    plan = np.array(result.x).reshape(len(supply), len(demand))
    df = pd.DataFrame(plan, 
                      index=['Warehouse A', 'Warehouse B'], 
                      columns=['Store X', 'Store Y', 'Store Z'])
    print(df.round(2))
    print(f"\nTotal Minimum Cost: ${result.fun:.2f}")
else:
    print("Optimization failed:", result.message)