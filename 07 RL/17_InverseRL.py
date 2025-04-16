"""
Project 257. Inverse reinforcement learning
Description:
Inverse Reinforcement Learning (IRL) flips traditional reinforcement learning on its head: instead of learning a policy from a reward function, we try to infer the reward function itself based on expert demonstrations. This is especially useful in fields like autonomous driving, robotics, and imitation learning, where defining the reward manually is difficult.

In this project, we implement a basic IRL framework using maximum margin planning in a simple GridWorld where we assume an expert provides optimal demonstrations.

About:
✅ What It Does:
Simulates an expert trajectory (optimal path).

Computes feature expectations from expert and policy rollouts.

Uses gradient ascent on feature margins to learn a reward function.

Visualizes the learned reward landscape, where the expert path is most rewarding.
"""


import numpy as np
import matplotlib.pyplot as plt
 
# Simple GridWorld (5x5)
class GridWorld:
    def __init__(self, size=5, goal=(4, 4)):
        self.size = size
        self.goal = goal
        self.states = [(i, j) for i in range(size) for j in range(size)]
 
    def features(self, state):
        # One-hot feature for each cell
        vec = np.zeros(self.size * self.size)
        idx = state[0] * self.size + state[1]
        vec[idx] = 1
        return vec
 
# Simulated expert policy (shortest path)
def generate_expert_trajectories(start, goal):
    traj = []
    pos = start
    while pos != goal:
        r, c = pos
        gr, gc = goal
        if r < gr:
            r += 1
        elif c < gc:
            c += 1
        pos = (r, c)
        traj.append(pos)
    return traj
 
# Feature expectations for a trajectory
def feature_expectation(traj, env):
    return sum([env.features(s) for s in traj]) / len(traj)
 
# Reward function: linear combination of features
def compute_reward(weights, env):
    rewards = {}
    for state in env.states:
        rewards[state] = np.dot(weights, env.features(state))
    return rewards
 
# Policy evaluation: simple greedy w.r.t reward
def generate_policy_from_reward(env, rewards, start, goal):
    state = start
    traj = []
    while state != goal:
        r, c = state
        candidates = []
        if r < env.size - 1:
            candidates.append((r+1, c))
        if c < env.size - 1:
            candidates.append((r, c+1))
        best = max(candidates, key=lambda s: rewards[s])
        state = best
        traj.append(state)
    return traj
 
# IRL loop (maximum margin imitation)
env = GridWorld()
start, goal = (0, 0), env.goal
expert_traj = generate_expert_trajectories(start, goal)
expert_feat = feature_expectation(expert_traj, env)
 
# Initialize random weights
weights = np.random.rand(env.size * env.size)
weights /= np.linalg.norm(weights)
 
history = []
 
for iteration in range(20):
    rewards = compute_reward(weights, env)
    policy_traj = generate_policy_from_reward(env, rewards, start, goal)
    policy_feat = feature_expectation(policy_traj, env)
 
    margin = expert_feat - policy_feat
    if np.linalg.norm(margin) < 1e-4:
        break
 
    weights += 0.1 * margin
    weights /= np.linalg.norm(weights)
    history.append(np.linalg.norm(margin))
 
print("✅ Inferred reward weights learned!")
 
# Visualize inferred reward map
reward_map = np.zeros((env.size, env.size))
for (r, c) in env.states:
    reward_map[r, c] = np.dot(weights, env.features((r, c)))
 
plt.imshow(reward_map, cmap="viridis")
plt.title("Inferred Reward Map (IRL)")
plt.colorbar()
plt.show()