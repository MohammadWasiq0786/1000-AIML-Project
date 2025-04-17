"""
Project 820. Energy Consumption Optimization

This project focuses on minimizing energy usage (and cost) while maintaining operational performance. In this example, we simulate how to allocate machine run-times or energy-consuming activities under constraints to minimize total energy consumption using linear programming.

This model finds the most energy-efficient distribution of workload across machines, respecting their capacity. It can be scaled up to optimize energy use across HVAC, lighting, or smart grid components in real-time.
"""

import numpy as np
from scipy.optimize import linprog
import pandas as pd
 
# Simulated scenario:
# 3 machines (M1, M2, M3) can complete a task using different energy levels per hour
# Our goal is to complete 100 units of work using minimum total energy
 
# Energy consumption per unit of work for each machine
energy_per_unit = [5, 4, 6]  # in kWh
 
# Max capacity (max units of work per machine)
capacity = [40, 35, 50]
 
# Objective: minimize total energy used
c = energy_per_unit  # minimize energy = sum(energy_per_unit * units_assigned)
 
# Constraints: sum of units worked by all machines = 100
A_eq = [[1, 1, 1]]
b_eq = [100]
 
# Bounds: work assigned to each machine must be within its capacity
bounds = [(0, cap) for cap in capacity]
 
# Solve the optimization problem
result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
 
# Output optimal energy plan
if result.success:
    allocation = result.x
    total_energy = result.fun
    df = pd.DataFrame({
        'Machine': ['M1', 'M2', 'M3'],
        'Units Assigned': allocation.round(2),
        'Energy per Unit': energy_per_unit,
        'Total Energy (kWh)': (allocation * energy_per_unit).round(2)
    })
    print("Optimal Energy Allocation Plan:")
    print(df)
    print(f"\nMinimum Total Energy Consumption: {total_energy:.2f} kWh")
else:
    print("Optimization failed:", result.message)