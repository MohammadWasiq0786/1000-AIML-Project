"""
Project 601: Multi-agent Reinforcement Learning
Description:
Multi-agent reinforcement learning (MARL) deals with scenarios where multiple agents interact in a shared environment. These agents may cooperate, compete, or be independent. In this project, we will explore multi-agent reinforcement learning and build a basic simulation where multiple agents learn and interact using reinforcement learning algorithms.
"""

import gym
import numpy as np
import random
 
# 1. Create a custom multi-agent environment using OpenAI Gym (e.g., multi-agent grid world)
class MultiAgentGridWorld(gym.Env):
    def __init__(self):
        super(MultiAgentGridWorld, self).__init__()
        self.grid_size = 5
        self.num_agents = 2
        self.action_space = gym.spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = gym.spaces.Discrete(self.grid_size * self.grid_size)
 
        self.agent_positions = [np.array([0, 0]), np.array([4, 4])]  # Initial positions for 2 agents
        self.goal_position = np.array([2, 2])  # Goal position in the grid
 
    def reset(self):
        self.agent_positions = [np.array([0, 0]), np.array([4, 4])]  # Reset agents to initial positions
        return self.agent_positions
 
    def step(self, actions):
        rewards = []
        for i, action in enumerate(actions):
            # Move agent based on the action
            if action == 0:  # Up
                self.agent_positions[i][1] += 1
            elif action == 1:  # Down
                self.agent_positions[i][1] -= 1
            elif action == 2:  # Left
                self.agent_positions[i][0] -= 1
            elif action == 3:  # Right
                self.agent_positions[i][0] += 1
 
            # Clip positions to stay within grid bounds
            self.agent_positions[i] = np.clip(self.agent_positions[i], 0, self.grid_size - 1)
 
            # Reward if agent reaches the goal
            if np.array_equal(self.agent_positions[i], self.goal_position):
                rewards.append(1)  # Goal reached
            else:
                rewards.append(0)  # No reward for not reaching goal
 
        done = all(np.array_equal(pos, self.goal_position) for pos in self.agent_positions)
        return self.agent_positions, rewards, done, {}
 
# 2. Create an environment
env = MultiAgentGridWorld()
 
# 3. Train multiple agents using Q-learning (for simplicity, we use a basic Q-learning algorithm for both agents)
q_tables = [np.zeros((env.grid_size * env.grid_size, env.action_space.n)) for _ in range(env.num_agents)]
 
# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
 
# 4. Training loop
for episode in range(1000):
    states = env.reset()  # Reset the environment
    done = False
    total_rewards = [0, 0]
 
    while not done:
        actions = []
        for i in range(env.num_agents):
            state = states[i][0] * env.grid_size + states[i][1]  # Convert (x, y) to state index
            if random.uniform(0, 1) < epsilon:  # Exploration
                action = env.action_space.sample()
            else:  # Exploitation
                action = np.argmax(q_tables[i][state])
 
            actions.append(action)
 
        next_states, rewards, done, _ = env.step(actions)
 
        # Update Q-values using Q-learning formula
        for i in range(env.num_agents):
            state = states[i][0] * env.grid_size + states[i][1]
            next_state = next_states[i][0] * env.grid_size + next_states[i][1]
            q_tables[i][state, actions[i]] += alpha * (rewards[i] + gamma * np.max(q_tables[i][next_state]) - q_tables[i][state, actions[i]])
 
        total_rewards = [total_rewards[i] + rewards[i] for i in range(env.num_agents)]
        states = next_states
 
    # Print episode info
    if episode % 100 == 0:
        print(f"Episode {episode}: Total Rewards for Agents: {total_rewards}")