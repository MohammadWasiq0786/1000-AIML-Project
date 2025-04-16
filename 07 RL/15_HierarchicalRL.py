"""
Project 255. Hierarchical reinforcement learning
Description:
Hierarchical Reinforcement Learning (HRL) breaks complex tasks into smaller sub-tasks using a manager-worker (meta-controller and controller) structure:

The high-level policy selects goals or sub-tasks.

The low-level policy handles execution to achieve that goal.

This structure improves scalability, reusability, and efficiency in long-horizon tasks. In this project, we'll simulate a basic HRL setup using a 2-level policy hierarchy in a custom GridWorld environment.

About:
✅ What It Does:
Simulates an HRL system with a high-level policy choosing subgoals.

Uses a low-level policy to reach those subgoals.

Demonstrates hierarchical control to navigate efficiently through the GridWorld.

Foundation for advanced tasks like robot task planning or multi-stage game objectives.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# Create a simple 5x5 GridWorld
class GridWorld:
    def __init__(self):
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.position = self.start
        self.subgoals = [(2, 2), (4, 4)]  # high-level manager picks one at a time
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R
 
    def reset(self):
        self.position = self.start
        return self.position
 
    def step(self, action):
        r, c = self.position
        dr, dc = self.actions[action]
        r = np.clip(r + dr, 0, self.size - 1)
        c = np.clip(c + dc, 0, self.size - 1)
        self.position = (r, c)
        reward = 1 if self.position == self.goal else -0.01
        done = self.position == self.goal
        return self.position, reward, done
 
# Low-level worker policy: go to subgoal
def low_level_policy(state, subgoal):
    sr, sc = state
    gr, gc = subgoal
    if sr < gr: return 1  # Down
    if sr > gr: return 0  # Up
    if sc < gc: return 3  # Right
    if sc > gc: return 2  # Left
    return None  # at subgoal
 
# High-level manager policy: pick subgoal
def high_level_policy(state):
    # First go to center, then to final goal
    if state != (2, 2):
        return (2, 2)
    else:
        return (4, 4)
 
# Run HRL episode
env = GridWorld()
state = env.reset()
trajectory = [state]
 
for step in range(100):
    subgoal = high_level_policy(state)
 
    while state != subgoal:
        action = low_level_policy(state, subgoal)
        if action is None: break
        state, _, _ = env.step(action)
        trajectory.append(state)
 
    if state == env.goal:
        break
 
# Visualize trajectory
grid = np.zeros((5, 5))
for (r, c) in trajectory:
    grid[r, c] = 1
 
plt.imshow(grid, cmap='Greens')
plt.title("HRL Path: Start to Subgoal(s) to Goal")
plt.xticks(range(5))
plt.yticks(range(5))
plt.grid(True)
plt.show()