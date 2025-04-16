"""
Project 242. Q-learning for grid world
Description:
Q-learning is a foundational model-free reinforcement learning algorithm that learns the value of actions in given states. In this project, we implement Q-learning in a simple Grid World, where the agent must learn to navigate from a starting point to a goal while avoiding traps. The agent updates a Q-table based on experience to find the optimal path.

About:
✅ What It Does:
Sets up a grid-based environment with a goal and negative traps.

Trains an agent using Q-learning to maximize cumulative reward.

Uses ε-greedy exploration for learning optimal policy.

Shows a visual map of the best action in each grid cell after training.
"""
import numpy as np
import matplotlib.pyplot as plt
 
# Grid world size and structure
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
TRAPS = [(1, 3), (2, 2), (3, 1)]  # Penalize these positions
 
# Q-table dimensions: (rows, cols, actions)
q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # Actions: 0=Up, 1=Down, 2=Left, 3=Right
 
# Parameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 0.1     # Exploration rate
episodes = 500
 
# Action mapping
actions = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}
 
# Get next state given an action
def step(state, action):
    row, col = state
    dr, dc = actions[action]
    new_row = np.clip(row + dr, 0, GRID_SIZE - 1)
    new_col = np.clip(col + dc, 0, GRID_SIZE - 1)
    new_state = (new_row, new_col)
 
    # Define rewards
    if new_state == GOAL:
        reward = 10
        done = True
    elif new_state in TRAPS:
        reward = -10
        done = True
    else:
        reward = -1
        done = False
 
    return new_state, reward, done
 
# Q-learning training loop
episode_rewards = []
 
for episode in range(episodes):
    state = START
    total_reward = 0
 
    while True:
        row, col = state
 
        # ε-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[row, col])
 
        next_state, reward, done = step(state, action)
        next_row, next_col = next_state
 
        # Q-learning update rule
        best_next_action = np.max(q_table[next_row, next_col])
        q_table[row, col, action] += alpha * (reward + gamma * best_next_action - q_table[row, col, action])
 
        state = next_state
        total_reward += reward
 
        if done:
            break
 
    episode_rewards.append(total_reward)
 
# Plot learning curve
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning in Grid World")
plt.grid(True)
plt.show()
 
# Display learned Q-values for each cell
print("\n📘 Learned Q-values (max action per state):")
for row in range(GRID_SIZE):
    line = ""
    for col in range(GRID_SIZE):
        best_action = np.argmax(q_table[row, col])
        arrows = ['↑', '↓', '←', '→']
        cell = arrows[best_action] if (row, col) != GOAL else 'G'
        line += f"{cell}  "
    print(line)