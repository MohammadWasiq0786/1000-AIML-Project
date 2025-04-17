"""
Project 809. Inventory Optimization System

An inventory optimization system ensures the right amount of stock is maintained to meet demand while minimizing holding and shortage costs. In this basic implementation, we simulate inventory levels, demand, and restocking using the Economic Order Quantity (EOQ) model—a classic approach in operations.

This model calculates the optimal order quantity that minimizes total inventory cost and simulates how inventory levels fluctuate over time with regular replenishment cycles.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# Input values for the EOQ model
annual_demand = 1200     # units per year
ordering_cost = 100      # cost to place one order
holding_cost = 2         # cost to hold one unit for a year
 
# EOQ formula: sqrt((2 * D * S) / H)
eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
 
# Calculate number of orders and total costs
num_orders = annual_demand / eoq
total_ordering_cost = num_orders * ordering_cost
avg_inventory = eoq / 2
total_holding_cost = avg_inventory * holding_cost
total_inventory_cost = total_ordering_cost + total_holding_cost
 
# Display results
print(f"Economic Order Quantity (EOQ): {eoq:.2f} units")
print(f"Number of Orders per Year: {num_orders:.1f}")
print(f"Total Ordering Cost: ${total_ordering_cost:.2f}")
print(f"Total Holding Cost: ${total_holding_cost:.2f}")
print(f"Total Inventory Cost: ${total_inventory_cost:.2f}")
 
# Visualize inventory level over time
cycle_days = 365 / num_orders
time = np.linspace(0, 365, 1000)
inventory = eoq - (eoq / cycle_days) * (time % cycle_days)
 
plt.figure(figsize=(10, 4))
plt.plot(time, inventory)
plt.title('Inventory Level Over Time (EOQ Model)')
plt.xlabel('Day of the Year')
plt.ylabel('Inventory Level')
plt.grid(True)
plt.tight_layout()
plt.show()