"""
Project 605: Offline Reinforcement Learning
Description:
Offline reinforcement learning (also known as batch RL) is the process of learning from a fixed dataset of interactions with the environment, without any further interaction during the training process. This is particularly useful when online learning is costly or impractical. In this project, we will implement offline RL using a pre-collected dataset of state-action-reward transitions to train an agent.
"""

import numpy as np
import gym
import random
 
# 1. Define a simple Q-learning agent for offline reinforcement learning
class OfflineQLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
 
    def select_action(self, state):
        # Epsilon-greedy action selection (offline, no exploration)
        if np.random.rand() < self.exploration_rate:
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
 
# 2. Simulate offline data collection (generate a fixed dataset from interacting with the environment)
env = gym.make('CartPole-v1')
 
# Offline data collection (simulating the environment's interactions)
offline_data = []
for episode in range(100):  # Collect data for 100 episodes
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Take a random action for exploration
        next_state, reward, done, _ = env.step(action)
        offline_data.append((state, action, reward, next_state, done))
        state = next_state
 
# 3. Create an offline RL agent
offline_agent = OfflineQLearningAgent(action_space=env.action_space.n)
 
# 4. Train the offline agent on the collected data (no interaction with the environment during training)
for episode in range(100):  # Train for 100 episodes
    random.shuffle(offline_data)  # Shuffle the offline data to avoid bias
    total_reward = 0
    for state, action, reward, next_state, done in offline_data:
        offline_agent.update_q_value(state, action, reward, next_state, done)
        total_reward += reward
    
    print(f"Offline RL Episode {episode + 1}, Total Reward: {total_reward}")
 
# 5. Evaluate the performance of the offline agent (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = offline_agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward for the offline agent in evaluation: {total_reward}")