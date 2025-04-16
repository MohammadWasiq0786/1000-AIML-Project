"""
Project 32. Simple reinforcement learning agent
Description:
Reinforcement Learning (RL) is a learning paradigm where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards. In this project, we implement a basic Q-learning agent to solve the classic FrozenLake environment using OpenAI Gym.

About:
🧠 What This Project Teaches:
How an agent learns by interacting with an environment

Balancing exploration vs. exploitation

Storing and updating Q-values for state-action pairs
"""

# Install OpenAI Gym if not already installed:
# pip install gym==0.26.2
 
import gym
import numpy as np
import random
 
# Initialize FrozenLake environment (4x4 grid, slippery by default)
env = gym.make("FrozenLake-v1", is_slippery=True)
 
# Q-table initialization: [state, action] = value
q_table = np.zeros((env.observation_space.n, env.action_space.n))
 
# Hyperparameters
alpha = 0.8        # Learning rate
gamma = 0.95       # Discount factor
epsilon = 1.0      # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 2000
max_steps = 100
 
# Training loop
for episode in range(episodes):
    state = env.reset()[0]
    done = False
 
    for _ in range(max_steps):
        # Choose action (explore or exploit)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
 
        # Take action
        new_state, reward, done, truncated, info = env.step(action)
 
        # Update Q-value
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
        )
 
        state = new_state
 
        if done:
            break
 
    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
 
# Print final Q-table
print("Trained Q-table:\n", q_table)
 
# Evaluate the agent
total_rewards = 0
for episode in range(100):
    state = env.reset()[0]
    done = False
    for _ in range(max_steps):
        action = np.argmax(q_table[state])
        new_state, reward, done, truncated, info = env.step(action)
        state = new_state
        if done:
            total_rewards += reward
            break
 
print(f"\nSuccess rate over 100 evaluation episodes: {total_rewards}%")