"""
Project 241. Multi-armed bandit implementation
Description:
The Multi-Armed Bandit (MAB) problem is a classic example of the exploration-exploitation trade-off in reinforcement learning. It simulates a slot machine (bandit) with multiple arms, each providing random rewards. The agent must learn which arm gives the highest reward over time by trying them and adjusting its strategy. We'll implement the ε-greedy algorithm, where with probability ε, the agent explores; otherwise, it exploits the best-known arm.

About:
✅ What It Does:
Simulates a slot machine with 10 arms.

Uses ε-greedy policy to choose between exploring and exploiting.

Learns which arm gives the best reward over time.

Visualizes learning progress as the average reward improves.
"""
import numpy as np
import matplotlib.pyplot as plt
 
# Define a Multi-Armed Bandit class
class MultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.true_means = np.random.normal(0, 1, num_arms)  # True reward for each arm
        self.counts = np.zeros(num_arms)                    # Times each arm was selected
        self.estimates = np.zeros(num_arms)                 # Estimated rewards
 
    def pull(self, arm):
        # Simulate pulling an arm: reward is drawn from N(true_mean, 1)
        return np.random.normal(self.true_means[arm], 1)
 
    def update(self, arm, reward):
        # Incremental average to update estimated reward
        self.counts[arm] += 1
        n = self.counts[arm]
        self.estimates[arm] += (1/n) * (reward - self.estimates[arm])
 
# ε-greedy strategy implementation
def run_bandit_epsilon_greedy(bandit, epsilon=0.1, steps=1000):
    rewards = np.zeros(steps)
    for step in range(steps):
        # ε chance to explore
        if np.random.rand() < epsilon:
            action = np.random.randint(bandit.num_arms)
        else:
            action = np.argmax(bandit.estimates)
 
        reward = bandit.pull(action)
        bandit.update(action, reward)
        rewards[step] = reward
 
    return rewards
 
# Run the simulation
num_arms = 10
bandit = MultiArmedBandit(num_arms)
rewards = run_bandit_epsilon_greedy(bandit, epsilon=0.1, steps=1000)
 
# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(rewards) / (np.arange(1, 1001)))
plt.title("Average Reward over Time (ε-greedy)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.grid(True)
plt.show()
 
# Print final estimates
print("\n🎰 True means of arms:")
print(np.round(bandit.true_means, 3))
print("\n📊 Estimated means after training:")
print(np.round(bandit.estimates, 3))