"""
Project 264. Hindsight experience replay
Description:
Hindsight Experience Replay (HER) is a clever trick for sparse-reward environments where agents rarely achieve the goal. The key idea:
👉 even if the agent failed its original goal, we pretend it was trying to reach where it actually ended up — and learn from that!
This makes every trajectory useful, turning failures into learning experiences.

In this project, we implement a simplified HER on a custom bit-flip goal-reaching task, where the agent must transform a bit vector into a target vector by flipping bits.

About:
✅ What It Does:
Simulates goal-conditioned tasks using a bit-flipping problem.

Augments experience by pretending alternative goals were intended.

Dramatically improves performance in sparse-reward settings.

Concept used in robot grasping, navigation, puzzle solving, etc.
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
 
# Simple BitFlip environment
class BitFlipEnv:
    def __init__(self, n_bits=5):
        self.n_bits = n_bits
        self.reset()
 
    def reset(self):
        self.state = np.random.randint(0, 2, self.n_bits)
        self.goal = np.random.randint(0, 2, self.n_bits)
        return self.state.copy(), self.goal.copy()
 
    def step(self, action):
        self.state[action] = 1 - self.state[action]  # flip bit
        done = np.array_equal(self.state, self.goal)
        reward = 1.0 if done else 0.0
        return self.state.copy(), reward, done
 
# Replay buffer with HER
class HERBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
 
    def add(self, episode):
        self.buffer.extend(episode)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
 
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, g, a, r, s2 = zip(*batch)
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(g),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(s2)
        )
 
# Neural network policy (Q-function)
class QNet(nn.Module):
    def __init__(self, n_bits):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits * 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_bits)
        )
 
    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        return self.net(x)
 
# Initialize
n_bits = 5
env = BitFlipEnv(n_bits)
buffer = HERBuffer()
qnet = QNet(n_bits)
optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
gamma = 0.98
episodes = 500
batch_size = 64
reward_curve = []
 
for ep in range(episodes):
    state, goal = env.reset()
    episode = []
 
    for t in range(n_bits * 2):
        s_tensor = torch.FloatTensor([state])
        g_tensor = torch.FloatTensor([goal])
        with torch.no_grad():
            q_values = qnet(s_tensor, g_tensor)
        action = q_values.argmax().item()
 
        # Random exploration
        if random.random() < 0.3:
            action = random.randint(0, n_bits - 1)
 
        next_state, reward, done = env.step(action)
        episode.append((state.copy(), goal.copy(), action, reward, next_state.copy()))
        state = next_state.copy()
        if done:
            break
 
    # Hindsight: pretend the goal was the final state
    final_state = next_state.copy()
    her_episode = []
    for (s, g, a, r, s2) in episode:
        her_episode.append((s, g, a, r, s2))
        her_goal = final_state
        her_reward = 1.0 if np.array_equal(s2, her_goal) else 0.0
        her_episode.append((s, her_goal, a, her_reward, s2))
 
    buffer.add(her_episode)
    reward_curve.append(int(done))
 
    # Learn from HER buffer
    if len(buffer.buffer) >= batch_size:
        s, g, a, r, s2 = buffer.sample(batch_size)
        q_values = qnet(s, g).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_q = qnet(s2, g).max(1)[0]
            target = r + gamma * max_q
        loss = nn.MSELoss()(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Episode {ep+1}, Solved: {int(done)}")
 
# Plot success rate
plt.plot(np.convolve(reward_curve, np.ones(10)/10, mode='valid'))
plt.title("HER Success Rate Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Success Rate (Smoothed)")
plt.grid(True)
plt.show()