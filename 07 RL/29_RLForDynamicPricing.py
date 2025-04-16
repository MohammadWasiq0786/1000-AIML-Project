"""
Project 269. RL for dynamic pricing
Description:
Dynamic pricing is a strategy where prices change over time based on demand, competition, or customer behavior. Using RL, we can train an agent to learn pricing strategies that maximize revenue or profit in response to customer reactions.

In this project, we simulate a simple customer pricing environment, where the agent learns how to adjust product prices to maximize revenue using a Q-learning approach.

About:
✅ What It Does:
Agent selects one of several price points per episode.

Reward depends on probability of purchase × price.

Learns optimal pricing to maximize expected revenue.

Foundation for real-world scenarios like airlines, e-commerce, ride-hailing, and subscriptions.
"""


import numpy as np
import matplotlib.pyplot as plt
import random
 
# Simulated environment for dynamic pricing
class PricingEnv:
    def __init__(self):
        self.price_points = [5, 10, 15, 20, 25]  # available prices
        self.state_space = 1  # stateless (could be extended)
        self.n_actions = len(self.price_points)
 
    def step(self, action):
        price = self.price_points[action]
 
        # Simulated customer behavior: demand decreases with price
        purchase_prob = max(0, 1 - price / 30.0)
        purchase = 1 if random.random() < purchase_prob else 0
 
        reward = price * purchase
        return 0, reward, True  # stateless, always done
 
    def reset(self):
        return 0  # dummy state
 
# Q-learning agent
class QAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = np.zeros((1, n_actions))  # stateless
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
 
    def select_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_table.shape[1] - 1)
        return np.argmax(self.q_table[0])
 
    def update(self, action, reward):
        current_q = self.q_table[0][action]
        new_q = current_q + self.alpha * (reward - current_q)
        self.q_table[0][action] = new_q
 
# Setup environment and agent
env = PricingEnv()
agent = QAgent(n_actions=env.n_actions)
 
episodes = 500
revenue_log = []
 
for ep in range(episodes):
    state = env.reset()
    action = agent.select_action()
    _, reward, _ = env.step(action)
    agent.update(action, reward)
 
    revenue_log.append(reward)
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Price: ${env.price_points[action]}, Revenue: ${reward:.2f}")
 
# Plot revenue trend
plt.plot(np.convolve(revenue_log, np.ones(10)/10, mode='valid'))
plt.title("RL for Dynamic Pricing – Revenue Over Time")
plt.xlabel("Episode")
plt.ylabel("Revenue (Smoothed)")
plt.grid(True)
plt.show()