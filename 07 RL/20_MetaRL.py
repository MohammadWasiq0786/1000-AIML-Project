"""
Project 260. Meta-reinforcement learning
Description:
Meta-Reinforcement Learning teaches agents how to learn new tasks quickly by learning from a distribution of tasks. It’s sometimes called “learning to learn”. The goal is to train a meta-policy that can adapt to new environments using very few experiences, mimicking how humans transfer prior knowledge.

In this project, we simulate meta-learning using RL^2-style encoding: we train a policy that learns not just from the current state, but also from past observations, actions, and rewards — giving it a memory of how to act in new tasks.

About:
✅ What It Does:
Trains a memory-augmented policy (via GRU) to adapt within a task.

Learns across multiple tasks — generalizes and transfers learning.

Demonstrates how meta-learning helps in few-shot adaptation.

Foundation for techniques like MAML, RL^2, and PEARL.
"""


# Install dependencies:
# pip install gym numpy torch matplotlib
 
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
 
# Create multiple variations of a simple bandit-like task
class OneStepBanditEnv:
    def __init__(self, reward_probs):
        self.reward_probs = reward_probs
        self.action_space = gym.spaces.Discrete(len(reward_probs))
 
    def reset(self):
        return np.zeros(1)  # Dummy state
 
    def step(self, action):
        reward = 1 if np.random.rand() < self.reward_probs[action] else 0
        return np.zeros(1), reward, True, {}, {}
 
# Memory-based meta-RL policy
class MetaPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
 
    def forward(self, inputs, hidden_state=None):
        rnn_out, h = self.rnn(inputs, hidden_state)
        logits = self.fc(rnn_out)
        return logits, h
 
# Sample a new task (i.e., new reward probabilities)
def sample_task():
    return OneStepBanditEnv(reward_probs=[random.random(), random.random()])
 
# Set up training
episodes = 3000
meta_policy = MetaPolicy(input_dim=3, hidden_dim=32, output_dim=2)
optimizer = optim.Adam(meta_policy.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
reward_history = []
 
for ep in range(episodes):
    env = sample_task()
    state = env.reset()
    hidden = None
    log = []
 
    # Use memory: input is [action_t-1, reward_t-1, timestep]
    prev_action = 0
    prev_reward = 0
    sequence = []
 
    total_reward = 0
 
    for t in range(5):  # few-shot rollout
        # Construct memory input
        timestep = t / 5
        inp = torch.FloatTensor([[prev_action, prev_reward, timestep]]).unsqueeze(0)
 
        logits, hidden = meta_policy(inp, hidden)
        action = torch.argmax(logits[0, 0]).item()
 
        _, reward, done, _, _ = env.step(action)
        total_reward += reward
 
        # Log data
        prev_action = float(action)
        prev_reward = float(reward)
 
    # Backprop through the entire sequence once
    # Train using randomly sampled rollouts (simulate imitation)
    y_true = torch.LongTensor([0 if env.reward_probs[0] > env.reward_probs[1] else 1])
    loss = loss_fn(logits[0], y_true)
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    reward_history.append(total_reward)
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Total Reward: {total_reward:.2f}")
 
# Plot reward improvement
plt.plot(reward_history)
plt.title("Meta-RL: Learning to Learn in Bandit Tasks")
plt.xlabel("Episode")
plt.ylabel("Total Reward per Task")
plt.grid(True)
plt.show()