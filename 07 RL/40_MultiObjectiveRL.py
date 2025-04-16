"""
Project 280. Multi-objective reinforcement learning
Description:
In real-world applications, agents often face multiple conflicting objectives — like speed vs safety, performance vs energy use, or profit vs fairness. Multi-Objective Reinforcement Learning (MORL) trains agents to balance trade-offs between these goals.

In this project, we simulate an agent that must complete a task quickly but is penalized for using too much power. The agent receives a vector reward with two components:

Progress reward – for getting closer to the goal

Energy penalty – for using high power actions

The agent learns to optimize a weighted sum of these objectives using Q-learning.

About:
✅ What It Does:
Agent moves toward a goal but must balance speed and energy use.

Learns from a vector reward, reduced to a scalar via user-defined weights.

Demonstrates reward trade-offs, which is essential in:

Robotics (speed vs battery life)

Finance (risk vs return)

Healthcare (effectiveness vs cost)
"""


import numpy as np
import matplotlib.pyplot as plt
import random
 
# Simulated environment: agent moves on a 1D line (0 to 10)
class MultiObjectiveEnv:
    def __init__(self):
        self.goal = 10
        self.reset()
 
    def reset(self):
        self.pos = 0
        return self.pos
 
    def step(self, action):
        # Action is step size ∈ {-2, -1, 0, 1, 2}
        self.pos += action
        self.pos = max(0, min(self.pos, self.goal))
 
        # Reward vector: [progress, energy_penalty]
        progress = 1 if self.pos == self.goal else 0
        energy_penalty = -abs(action) * 0.1
        reward = (progress, energy_penalty)
 
        done = self.pos == self.goal
        return self.pos, reward, done
 
# Q-learning agent with reward weighting
class MultiObjectiveAgent:
    def __init__(self, n_states=11, actions=[-2, -1, 0, 1, 2], weights=(1.0, 0.5)):
        self.q_table = np.zeros((n_states, len(actions)))
        self.actions = actions
        self.weights = weights  # [progress_weight, energy_weight]
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
 
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        idx = np.argmax(self.q_table[state])
        return self.actions[idx]
 
    def update(self, state, action, reward_vec, next_state):
        reward = self.weights[0] * reward_vec[0] + self.weights[1] * reward_vec[1]
        a_idx = self.actions.index(action)
        max_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * max_next
        self.q_table[state][a_idx] += self.alpha * (td_target - self.q_table[state][a_idx])
 
# Train agent
env = MultiObjectiveEnv()
agent = MultiObjectiveAgent()
episodes = 300
reward_log = []
 
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
 
    while not done:
        action = agent.choose_action(state)
        next_state, reward_vec, done = env.step(action)
        agent.update(state, action, reward_vec, next_state)
        state = next_state
        total_reward += reward_vec[0] + reward_vec[1]
 
    reward_log.append(total_reward)
    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}, Weighted Total Reward: {total_reward:.2f}")
 
# Plot performance
plt.plot(np.convolve(reward_log, np.ones(10)/10, mode='valid'))
plt.title("Multi-Objective RL – Weighted Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Weighted Reward")
plt.grid(True)
plt.show()