"""
Project 254. On-policy vs off-policy learning
Description:
One of the fundamental distinctions in reinforcement learning is between on-policy and off-policy methods:

On-Policy: The agent learns from the policy it is currently following (e.g., SARSA).

Off-Policy: The agent learns about one policy (target) while following another (behavior) (e.g., Q-learning).

In this project, we'll implement both SARSA (on-policy) and Q-learning (off-policy) side-by-side in the FrozenLake-v1 environment and compare their value functions.

About:
✅ What It Does:
Implements SARSA (on-policy): learns from actions actually taken.

Implements Q-Learning (off-policy): learns from the best possible action.

Visualizes how different policies shape the learned value function.

Highlights how SARSA may be more conservative, while Q-learning is more optimistic.
"""

# Install dependencies:
# pip install gym numpy matplotlib
 
import gym
import numpy as np
import matplotlib.pyplot as plt
 
env = gym.make("FrozenLake-v1", is_slippery=False)  # deterministic
 
n_states = env.observation_space.n
n_actions = env.action_space.n
 
# Initialize Q-tables for SARSA (on-policy) and Q-learning (off-policy)
Q_sarsa = np.zeros((n_states, n_actions))
Q_qlearn = np.zeros((n_states, n_actions))
 
# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 1000
 
def epsilon_greedy(Q, state):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(Q[state])
 
# SARSA (On-policy)
for _ in range(episodes):
    state = env.reset()[0]
    action = epsilon_greedy(Q_sarsa, state)
    done = False
 
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = epsilon_greedy(Q_sarsa, next_state)
        td_target = reward + gamma * Q_sarsa[next_state][next_action]
        Q_sarsa[state][action] += alpha * (td_target - Q_sarsa[state][action])
        state, action = next_state, next_action
 
# Q-learning (Off-policy)
for _ in range(episodes):
    state = env.reset()[0]
    done = False
 
    while not done:
        action = epsilon_greedy(Q_qlearn, state)
        next_state, reward, done, _, _ = env.step(action)
        best_next = np.max(Q_qlearn[next_state])
        td_target = reward + gamma * best_next
        Q_qlearn[state][action] += alpha * (td_target - Q_qlearn[state][action])
        state = next_state
 
# Compare the Value Functions (max Q per state)
V_sarsa = np.max(Q_sarsa, axis=1).reshape((4, 4))
V_qlearn = np.max(Q_qlearn, axis=1).reshape((4, 4))
 
# Plotting
plt.figure(figsize=(10, 4))
 
plt.subplot(1, 2, 1)
plt.imshow(V_sarsa, cmap='Blues')
plt.title("SARSA (On-Policy)")
plt.colorbar()
 
plt.subplot(1, 2, 2)
plt.imshow(V_qlearn, cmap='Oranges')
plt.title("Q-Learning (Off-Policy)")
plt.colorbar()
 
plt.suptitle("On-Policy vs Off-Policy Value Functions (FrozenLake)")
plt.tight_layout()
plt.show()