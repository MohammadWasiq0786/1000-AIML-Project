"""
Project 261. Distributional reinforcement learning
Description:
Distributional Reinforcement Learning goes beyond estimating the expected return (value function) and instead learns the entire distribution of possible returns from a given state-action pair. This helps the agent better model uncertainty and variability in outcomes, enabling more robust decision-making.

In this project, we implement a basic Categorical DQN-style approach, where we approximate the return distribution using a fixed set of discrete bins (atoms), inspired by C51 algorithm.

About:
✅ What It Does:
Learns a probability distribution over returns (not just expected value).

Uses categorical projection to approximate distributional Bellman update.

Helps agents handle uncertainty and risk sensitivity in decision-making.

Lays groundwork for advanced dist-RL methods like C51, QR-DQN, and Implicit Q-Learning.
"""


import numpy as np
import matplotlib.pyplot as plt
import random
 
# Define a simple GridWorld (3x3)
class MiniGrid:
    def __init__(self):
        self.size = 3
        self.goal = (2, 2)
        self.reset()
 
    def reset(self):
        self.agent = (0, 0)
        return self.agent
 
    def step(self, action):
        moves = [(-1,0), (1,0), (0,-1), (0,1)]  # U, D, L, R
        r, c = self.agent
        dr, dc = moves[action]
        r = np.clip(r + dr, 0, self.size-1)
        c = np.clip(c + dc, 0, self.size-1)
        self.agent = (r, c)
        reward = 1 if self.agent == self.goal else 0
        done = self.agent == self.goal
        return self.agent, reward, done
 
    def state_to_index(self, pos):
        return pos[0] * self.size + pos[1]
 
# Initialize
env = MiniGrid()
n_states = env.size * env.size
n_actions = 4
atoms = 11
Vmin, Vmax = 0, 1
delta_z = (Vmax - Vmin) / (atoms - 1)
z = np.linspace(Vmin, Vmax, atoms)
 
# Initialize distributional Q-table
Q_dist = np.ones((n_states, n_actions, atoms)) / atoms  # uniform init
 
# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 300
 
# Training loop
for ep in range(episodes):
    state = env.reset()
    done = False
 
    while not done:
        s_idx = env.state_to_index(state)
 
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            q_values = (Q_dist[s_idx] * z).sum(axis=1)
            action = np.argmax(q_values)
 
        next_state, reward, done = env.step(action)
        ns_idx = env.state_to_index(next_state)
 
        # Distributional Bellman update
        p = np.zeros(atoms)
        for j in range(atoms):
            Tz = reward + gamma * z[j]
            Tz = np.clip(Tz, Vmin, Vmax)
            bj = (Tz - Vmin) / delta_z
            l, u = int(np.floor(bj)), int(np.ceil(bj))
 
            if l == u:
                p[l] += Q_dist[ns_idx][np.argmax((Q_dist[ns_idx] * z).sum(axis=1))][j]
            else:
                p[l] += Q_dist[ns_idx][np.argmax((Q_dist[ns_idx] * z).sum(axis=1))][j] * (u - bj)
                p[u] += Q_dist[ns_idx][np.argmax((Q_dist[ns_idx] * z).sum(axis=1))][j] * (bj - l)
 
        # Soft update (blend with current distribution)
        Q_dist[s_idx][action] = (1 - alpha) * Q_dist[s_idx][action] + alpha * p
        state = next_state
 
# Visualize learned return distributions for one state
plt.figure(figsize=(8, 5))
state_index = env.state_to_index((0, 0))
for a in range(n_actions):
    plt.plot(z, Q_dist[state_index][a], label=f"Action {a}")
plt.title("Learned Return Distributions for State (0, 0)")
plt.xlabel("Return")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()