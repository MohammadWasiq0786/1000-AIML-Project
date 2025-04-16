"""
Project 34. Multi-armed bandit implementation
Description:
The Multi-Armed Bandit problem is a classic example of the exploration vs. exploitation trade-off. It models situations where a decision-maker must choose between multiple options (arms) with uncertain rewards, aiming to maximize total reward over time. In this project, we implement the ε-greedy strategy to solve a multi-armed bandit problem with simulated slot machines.

About:
🎰 What This Project Demonstrates:
Simulates multiple slot machines (arms)

Uses the ε-greedy strategy to balance exploration and exploitation

Updates reward estimates incrementally

Tracks total reward and arm pulls
"""


import numpy as np
import matplotlib.pyplot as plt
 
# Simulated slot machines with different true success probabilities
true_conversion_rates = [0.2, 0.5, 0.75]
n_arms = len(true_conversion_rates)
n_rounds = 1000
epsilon = 0.1  # Exploration probability
 
# Initialize counts and estimated rewards
counts = np.zeros(n_arms)           # How many times each arm has been pulled
estimated_rewards = np.zeros(n_arms)  # Estimated value of each arm
total_reward = 0
reward_history = []
 
# ε-greedy algorithm loop
for t in range(n_rounds):
    # Exploration vs. Exploitation
    if np.random.rand() < epsilon:
        chosen_arm = np.random.randint(n_arms)  # Explore
    else:
        chosen_arm = np.argmax(estimated_rewards)  # Exploit best so far
 
    # Simulate pulling the arm
    reward = np.random.binomial(1, true_conversion_rates[chosen_arm])
    counts[chosen_arm] += 1
 
    # Update estimated reward using incremental average
    estimated_rewards[chosen_arm] += (reward - estimated_rewards[chosen_arm]) / counts[chosen_arm]
    total_reward += reward
    reward_history.append(total_reward)
 
# Results
print("True Conversion Rates:", true_conversion_rates)
print("Estimated Rewards:", estimated_rewards.round(3))
print("Arm Pull Counts:", counts.astype(int))
print(f"Total Reward Collected: {total_reward}")
 
# Plot reward growth
plt.figure(figsize=(8, 5))
plt.plot(reward_history)
plt.title("Total Reward Over Time (ε-Greedy)")
plt.xlabel("Round")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.tight_layout()
plt.show()