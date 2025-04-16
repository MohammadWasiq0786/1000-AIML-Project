"""
Project 256. Multi-agent reinforcement learning
Description:
Multi-Agent Reinforcement Learning (MARL) involves multiple agents learning and interacting in the same environment — either cooperatively, competitively, or both. Each agent may have its own policy, rewards, and learning objectives. In this project, we simulate a simple competitive MARL environment: two agents race to a goal in a 5×5 grid. The first one to reach the goal wins, the other gets nothing.

About:
✅ What It Does:
Simulates 2 agents racing to reach the same goal first.

Uses greedy policies to plan toward the goal.

Visually shows path conflicts and coordination (or lack thereof).

Lays the groundwork for more advanced MARL like Cooperative Navigation, Pong Duel, or Self-Play.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# GridWorld environment for two agents
class MultiAgentGrid:
    def __init__(self, size=5, goal=(4, 4)):
        self.size = size
        self.goal = goal
        self.reset()
 
    def reset(self):
        self.agent1 = (0, 0)
        self.agent2 = (0, self.size - 1)
        self.done = False
        return self.agent1, self.agent2
 
    def step(self, a1_action, a2_action):
        def move(pos, action):
            actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            r, c = pos
            dr, dc = actions[action]
            r = np.clip(r + dr, 0, self.size - 1)
            c = np.clip(c + dc, 0, self.size - 1)
            return (r, c)
 
        self.agent1 = move(self.agent1, a1_action)
        self.agent2 = move(self.agent2, a2_action)
 
        reward1 = 1 if self.agent1 == self.goal else 0
        reward2 = 1 if self.agent2 == self.goal else 0
 
        self.done = self.agent1 == self.goal or self.agent2 == self.goal
        return self.agent1, self.agent2, reward1, reward2, self.done
 
# Simple greedy policies: go to goal
def greedy_policy(pos, goal):
    pr, pc = pos
    gr, gc = goal
    if pr < gr: return 1  # Down
    if pr > gr: return 0  # Up
    if pc < gc: return 3  # Right
    if pc > gc: return 2  # Left
    return 0  # stay
 
# Initialize environment
env = MultiAgentGrid()
trajectory = []
 
# Simulate game
state = env.reset()
trajectory.append(state)
 
for _ in range(20):
    a1_action = greedy_policy(env.agent1, env.goal)
    a2_action = greedy_policy(env.agent2, env.goal)
 
    state = env.step(a1_action, a2_action)
    trajectory.append((env.agent1, env.agent2))
 
    if state[-1]:  # if done
        break
 
# Visualize paths
grid = np.zeros((env.size, env.size, 3))  # RGB for both agents
for step, (a1, a2) in enumerate(trajectory):
    r1, c1 = a1
    r2, c2 = a2
    grid[r1, c1, 0] = 1  # Red channel for agent 1
    grid[r2, c2, 1] = 1  # Green channel for agent 2
 
plt.imshow(grid)
plt.title("Multi-Agent Grid Race: Red vs Green")
plt.xticks(range(env.size))
plt.yticks(range(env.size))
plt.grid(True)
plt.show()