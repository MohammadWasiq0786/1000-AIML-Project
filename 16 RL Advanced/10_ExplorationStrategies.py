"""
Project 610: Exploration Strategies in RL
Description:
Exploration strategies in reinforcement learning (RL) are techniques that help the agent explore the environment more effectively to discover optimal actions. Common exploration strategies include epsilon-greedy, Boltzmann exploration, and Thompson sampling. In this project, we will implement a basic exploration strategy like epsilon-greedy and discuss how it can be applied to improve learning efficiency in RL.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define a simple Q-learning agent with epsilon-greedy exploration strategy
class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
 
    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table.get(state, np.zeros(len(self.action_space))))
 
    def update_q_value(self, state, action, reward, next_state, done):
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table.get(next_state, np.zeros(len(self.action_space))))
        current_q_value = self.q_table.get(state, np.zeros(len(self.action_space)))[action]
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table.get(next_state, np.zeros(len(self.action_space)))[best_next_action] - current_q_value)
 
        # Update Q-table
        self.q_table.setdefault(state, np.zeros(len(self.action_space)))[action] = new_q_value
 
        # Decay epsilon (exploration rate)
        if done:
            self.epsilon *= self.epsilon_decay
 
# 2. Initialize the environment and Q-learning agent
env = gym.make('CartPole-v1')
agent = QLearningAgent(action_space=env.action_space.n)
 
# 3. Train the agent using epsilon-greedy exploration
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
 
print(f"Total reward after training (epsilon-greedy): {total_reward}")