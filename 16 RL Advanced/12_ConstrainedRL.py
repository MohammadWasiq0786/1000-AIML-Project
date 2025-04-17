"""
Project 612: Constrained Reinforcement Learning
Description:
Constrained reinforcement learning (CRL) is a framework where the agent is tasked with not only maximizing the reward but also satisfying certain constraints or safety requirements during training. These constraints could be related to physical limits (e.g., staying within a specific range) or other safety factors. In this project, we will implement CRL where the agent learns to maximize rewards while respecting predefined constraints.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define a simple Q-learning agent with constraints
class ConstrainedQLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, constraint_threshold=0.5):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.constraint_threshold = constraint_threshold  # Define the constraint (e.g., max position)
        self.q_table = {}
 
    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table.get(state, np.zeros(len(self.action_space))))
 
    def update_q_value(self, state, action, reward, next_state, done):
        # Constrained reward shaping: Penalize if the agent violates the constraint
        if abs(state[0]) > self.constraint_threshold:  # Example constraint (position exceeds threshold)
            reward -= 2  # Penalize if constraint is violated
 
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table.get(next_state, np.zeros(len(self.action_space))))
        current_q_value = self.q_table.get(state, np.zeros(len(self.action_space)))[action]
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table.get(next_state, np.zeros(len(self.action_space)))[best_next_action] - current_q_value)
 
        # Update Q-table
        self.q_table.setdefault(state, np.zeros(len(self.action_space)))[action] = new_q_value
 
        # Decay epsilon (exploration rate)
        if done:
            self.epsilon *= self.epsilon_decay
 
# 2. Initialize the environment and Constrained Q-learning agent
env = gym.make('CartPole-v1')
agent = ConstrainedQLearningAgent(action_space=env.action_space.n)
 
# 3. Train the agent using Constrained Q-learning
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
 
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.update_q_value(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
 
# 4. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after Constrained RL training: {total_reward}")