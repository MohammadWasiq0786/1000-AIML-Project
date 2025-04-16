"""
Project 268. RL for recommendation systems
Description:
In recommendation systems, RL can go beyond static models by learning from user interaction over time. Instead of just predicting what a user might like, RL agents explore and adapt, learning what to recommend next based on feedback — clicks, watch time, likes, etc.

In this project, we simulate a simple user model and train a multi-armed bandit-style RL agent to recommend items and adapt based on user preferences.

About:
✅ What It Does:
Simulates a bandit-like recommendation scenario.

User has hidden preference probabilities per item.

Agent learns via ε-greedy exploration and updates Q-values.

Demonstrates interactive learning — adapts to maximize user engagement over time.
"""


import numpy as np
import matplotlib.pyplot as plt
import random
 
# Simulated user model with hidden preferences
class SimulatedUser:
    def __init__(self, n_items=5):
        self.n_items = n_items
        self.preferences = np.random.rand(n_items)  # unknown to agent
 
    def respond(self, item):
        # User has higher chance to click on preferred items
        click_prob = self.preferences[item]
        return 1 if random.random() < click_prob else 0
 
# Simple ε-greedy RL agent
class RecommenderAgent:
    def __init__(self, n_items, epsilon=0.1, alpha=0.1):
        self.n_items = n_items
        self.q_values = np.zeros(n_items)
        self.epsilon = epsilon
        self.alpha = alpha
 
    def select_item(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_items - 1)
        return np.argmax(self.q_values)
 
    def update(self, item, reward):
        self.q_values[item] += self.alpha * (reward - self.q_values[item])
 
# Initialize environment
n_items = 5
user = SimulatedUser(n_items)
agent = RecommenderAgent(n_items)
 
# Simulation
episodes = 500
clicks = []
cumulative_reward = 0
 
for ep in range(episodes):
    item = agent.select_item()
    reward = user.respond(item)
    agent.update(item, reward)
 
    cumulative_reward += reward
    clicks.append(cumulative_reward)
 
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Cumulative Clicks: {cumulative_reward}")
 
# Plot learning curve
plt.plot(clicks)
plt.title("RL-Based Recommender: Cumulative Clicks Over Time")
plt.xlabel("Episode")
plt.ylabel("Cumulative Clicks")
plt.grid(True)
plt.show()