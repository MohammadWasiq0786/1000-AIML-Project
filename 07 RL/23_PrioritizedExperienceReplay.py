"""
Project 263. Prioritized experience replay
Description:
In traditional experience replay, experiences are sampled uniformly at random. But in Prioritized Experience Replay (PER), we sample important transitions more frequently — those with higher TD-error — because they’re more informative to the agent’s learning.

This leads to faster convergence and better performance. We’ll implement a basic proportional PER using TD-error as priority and sample based on it.

About:
✅ What It Does:
Implements a custom replay buffer that samples based on TD-error.

Learns more efficiently by focusing on high-error, high-leverage experiences.

Uses importance sampling to correct bias introduced by prioritization.

Easy to integrate into DQN and other deep RL pipelines.
"""


import gym
import numpy as np
import random
import matplotlib.pyplot as plt
 
# FrozenLake for simplicity
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
 
# Q-table
Q = np.zeros((n_states, n_actions))
 
# Prioritized Replay Buffer
class PrioritizedReplay:
    def __init__(self, capacity=1000, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
 
    def add(self, experience, td_error):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(experience)
        self.priorities.append(max_priority)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
            self.priorities.pop(0)
 
    def sample(self, batch_size, beta=0.4):
        scaled_p = np.array(self.priorities) ** self.alpha
        probs = scaled_p / sum(scaled_p)
 
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
 
        # Importance-sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights
 
    def update_priority(self, index, td_error):
        self.priorities[index] = abs(td_error) + 1e-5  # avoid 0 priority
 
# Hyperparameters
episodes = 500
alpha = 0.1
gamma = 0.99
epsilon = 0.1
batch_size = 32
 
replay = PrioritizedReplay()
rewards = []
 
for ep in range(episodes):
    state = env.reset()[0]
    done = False
    ep_reward = 0
 
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
 
        next_state, reward, done, _, _ = env.step(action)
        td_error = reward + gamma * np.max(Q[next_state]) - Q[state][action]
 
        replay.add((state, action, reward, next_state, done), td_error)
 
        # Learn from prioritized replay
        if len(replay.buffer) >= batch_size:
            samples, indices, weights = replay.sample(batch_size)
            for i, (s, a, r, s2, d) in enumerate(samples):
                target = r + gamma * np.max(Q[s2]) * (1 - d)
                td = target - Q[s][a]
                Q[s][a] += alpha * weights[i] * td
                replay.update_priority(indices[i], td)
 
        state = next_state
        ep_reward += reward
 
    rewards.append(ep_reward)
    print(f"Episode {ep+1}, Reward: {ep_reward}")
 
# Plot reward trend
plt.plot(rewards)
plt.title("Q-Learning with Prioritized Experience Replay")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()