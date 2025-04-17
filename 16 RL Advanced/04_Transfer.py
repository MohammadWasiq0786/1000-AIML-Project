"""
Project 604: Transfer Reinforcement Learning
Description:
Transfer reinforcement learning (TRL) involves transferring knowledge learned from one task (source task) to improve the learning process in a different but related task (target task). The goal is to help the agent generalize knowledge across tasks without starting from scratch each time. In this project, we will explore how to apply transfer learning in reinforcement learning using pre-trained policies or value functions to speed up the learning process in a new environment.
"""

import numpy as np
import gym
 
# 1. Define a simple Q-learning agent
class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}  # Q-table to store state-action values
 
    def select_action(self, state):
        # Epsilon-greedy action selection
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
 
        # Decay exploration rate
        if done:
            self.exploration_rate *= self.exploration_decay
 
# 2. Transfer learning in Q-learning: Transfer Q-values from source task to target task
def transfer_learning(source_agent, target_agent):
    # Transfer Q-table from source agent to target agent
    target_agent.q_table = source_agent.q_table.copy()
 
# 3. Training process on source task (CartPole-v1)
source_env = gym.make('CartPole-v1')
source_agent = QLearningAgent(action_space=source_env.action_space.n)
 
# Train the source agent
for episode in range(1000):
    state = source_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = source_agent.select_action(state)
        next_state, reward, done, _ = source_env.step(action)
        source_agent.update_q_value(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    if episode % 100 == 0:
        print(f"Source Task Episode {episode}, Total Reward: {total_reward}")
 
# 4. Transfer the knowledge to a new task (MountainCar-v0)
target_env = gym.make('MountainCar-v0')
target_agent = QLearningAgent(action_space=target_env.action_space.n)
 
# Transfer Q-values from source agent to target agent
transfer_learning(source_agent, target_agent)
 
# Train the target agent using transferred knowledge
for episode in range(1000):
    state = target_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = target_agent.select_action(state)
        next_state, reward, done, _ = target_env.step(action)
        target_agent.update_q_value(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    if episode % 100 == 0:
        print(f"Target Task Episode {episode}, Total Reward: {total_reward}")