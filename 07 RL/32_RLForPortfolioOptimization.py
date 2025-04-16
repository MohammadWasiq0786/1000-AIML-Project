"""
Project 272. RL for portfolio optimization
Description:
In finance, portfolio optimization involves allocating capital across assets to maximize return and minimize risk. RL can be applied to learn dynamic investment strategies that adapt to market conditions over time. The agent receives price changes and decides how to rebalance its portfolio at each step.

In this project, we simulate a simple 2-asset market where an RL agent learns to adjust investment weights based on historical price movements to optimize cumulative return.

About:
✅ What It Does:
Simulates a market with two assets (different volatilities).

Learns how to reallocate capital to maximize returns.

Uses Q-learning with discretized state space (price relatives).

Foundation for applying RL to real stock data, crypto portfolios, or multi-asset ETFs.
"""


import numpy as np
import matplotlib.pyplot as plt
import random
 
# Simulated 2-asset market
class PortfolioEnv:
    def __init__(self, episodes=1000):
        self.episodes = episodes
        self.prices = self.generate_prices()
        self.reset()
 
    def generate_prices(self):
        # Generate log-normal random walk for 2 assets
        steps = 200
        p1 = np.cumprod(np.random.normal(1.0005, 0.01, steps))
        p2 = np.cumprod(np.random.normal(1.0002, 0.02, steps))
        return np.vstack([p1, p2]).T  # shape (T, 2)
 
    def reset(self):
        self.t = 0
        self.weights = np.array([0.5, 0.5])
        self.initial_value = 1.0
        return self.get_state()
 
    def get_state(self):
        price_rel = self.prices[self.t + 1] / self.prices[self.t]
        return price_rel
 
    def step(self, action):  # action ∈ {0, 1, 2}
        if action == 0:
            self.weights = np.array([0.0, 1.0])
        elif action == 1:
            self.weights = np.array([1.0, 0.0])
        else:
            self.weights = np.array([0.5, 0.5])
 
        self.t += 1
        if self.t >= len(self.prices) - 1:
            done = True
        else:
            done = False
 
        state = self.get_state()
        reward = np.dot(self.weights, state) - 1  # portfolio return at step
        return state, reward, done
 
# Q-learning agent
class PortfolioAgent:
    def __init__(self, n_actions=3):
        self.q_table = np.zeros((10, 10, n_actions))  # discretize states
        self.epsilon = 0.2
        self.alpha = 0.1
        self.gamma = 0.9
 
    def discretize(self, state):
        s1 = min(int(state[0] * 10), 9)
        s2 = min(int(state[1] * 10), 9)
        return s1, s2
 
    def choose_action(self, state):
        s1, s2 = self.discretize(state)
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        return np.argmax(self.q_table[s1][s2])
 
    def update(self, state, action, reward, next_state):
        s1, s2 = self.discretize(state)
        ns1, ns2 = self.discretize(next_state)
        best_next = np.max(self.q_table[ns1][ns2])
        td_target = reward + self.gamma * best_next
        self.q_table[s1][s2][action] += self.alpha * (td_target - self.q_table[s1][s2][action])
 
# Training
env = PortfolioEnv()
agent = PortfolioAgent()
returns = []
 
for ep in range(1000):
    state = env.reset()
    total_return = 1.0
    done = False
 
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_return *= 1 + reward  # accumulate wealth
 
    returns.append(total_return)
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Final Portfolio Value: {total_return:.4f}")
 
# Plot final wealth over episodes
plt.plot(np.convolve(returns, np.ones(10)/10, mode='valid'))
plt.title("RL for Portfolio Optimization")
plt.xlabel("Episode")
plt.ylabel("Final Portfolio Value")
plt.grid(True)
plt.show()