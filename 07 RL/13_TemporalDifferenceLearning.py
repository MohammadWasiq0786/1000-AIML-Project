"""
Project 253. Temporal difference learning
Description:
Temporal Difference (TD) Learning is a core RL technique that learns value functions from partial experiences without waiting for the final outcome. It combines ideas from Monte Carlo methods and dynamic programming. TD(0), the simplest form, updates the value of a state based on the next state’s estimated value, making it suitable for online and incremental learning.

In this project, we implement TD(0) for state value estimation in the FrozenLake-v1 environment using OpenAI Gym.

About:
✅ What It Does:
Updates state values step-by-step using TD(0).

Learns directly from the experience trajectory, not waiting for episode end.

Trains a value function even with stochastic environments (if is_slippery=True).

Forms the foundation for Q-learning, SARSA, and TD(λ).
"""


# Install dependencies:
# pip install gym numpy matplotlib
 
import gym
import numpy as np
import matplotlib.pyplot as plt
 
# Initialize FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)  # deterministic version
n_states = env.observation_space.n
n_actions = env.action_space.n
 
# Initialize value function
V = np.zeros(n_states)
 
# Hyperparameters
alpha = 0.1       # learning rate
gamma = 0.99      # discount factor
episodes = 1000
 
# Policy: simple greedy (not learning policy here)
def policy(state):
    return env.action_space.sample()  # Random for this example
 
# TD(0) learning loop
for episode in range(episodes):
    state = env.reset()[0]
    done = False
 
    while not done:
        action = policy(state)
        next_state, reward, done, _, _ = env.step(action)
 
        # TD(0) update
        V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
        state = next_state
 
# Display learned state values
print("📊 Estimated State Values (TD(0)):\n")
for s in range(n_states):
    print(f"State {s}: {V[s]:.4f}")
 
# Reshape for better visual (FrozenLake is 4x4)
V_grid = V.reshape((4, 4))
plt.imshow(V_grid, cmap='cool')
plt.colorbar()
plt.title("State Value Function (TD(0)) - FrozenLake")
plt.show()