"""
Project 273. RL for traffic light control
Description:
Traffic congestion is a huge urban challenge. RL can be used to optimize traffic light timings to reduce wait times, queue lengths, and travel time. By observing traffic flow and adjusting light phases, agents learn dynamic control policies that outperform static timers.

In this project, we simulate a simple intersection with cars arriving randomly and train a Q-learning agent to switch the traffic light intelligently.

About:
✅ What It Does:
Simulates dynamic car arrivals and a 2-phase traffic light.

Trains an RL agent to switch lights based on real-time traffic.

Minimizes total vehicle waiting using Q-learning.

Extensible to multi-intersection control and real-world sensor input.
"""


import numpy as np
import matplotlib.pyplot as plt
import random
 
# Simple traffic intersection with 2 directions: North-South (NS) and East-West (EW)
class TrafficEnv:
    def __init__(self):
        self.max_queue = 10
        self.reset()
 
    def reset(self):
        self.ns_queue = random.randint(0, 5)
        self.ew_queue = random.randint(0, 5)
        self.light = 0  # 0: NS green, 1: EW green
        return self.get_state()
 
    def get_state(self):
        return (self.ns_queue, self.ew_queue, self.light)
 
    def step(self, action):
        # Action: 0 = keep current light, 1 = switch light
        if action == 1:
            self.light = 1 - self.light
 
        # Vehicles pass based on green light
        if self.light == 0:  # NS green
            passed = min(2, self.ns_queue)
            self.ns_queue -= passed
        else:  # EW green
            passed = min(2, self.ew_queue)
            self.ew_queue -= passed
 
        # Random arrivals
        self.ns_queue += np.random.binomial(1, 0.6)
        self.ew_queue += np.random.binomial(1, 0.6)
 
        # Limit queue size
        self.ns_queue = min(self.ns_queue, self.max_queue)
        self.ew_queue = min(self.ew_queue, self.max_queue)
 
        # Reward = negative total queue length (we want to minimize waiting)
        reward = - (self.ns_queue + self.ew_queue)
        return self.get_state(), reward, False  # never ends (episodic)
 
# Q-learning agent
class TrafficAgent:
    def __init__(self):
        self.q_table = np.zeros((11, 11, 2, 2))  # ns_queue, ew_queue, light, action
        self.epsilon = 0.2
        self.alpha = 0.1
        self.gamma = 0.95
 
    def choose_action(self, state):
        ns, ew, light = state
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return np.argmin(self.q_table[ns][ew][light])  # minimizing queue
 
    def update(self, state, action, reward, next_state):
        ns, ew, light = state
        n_ns, n_ew, n_light = next_state
        best_next = np.min(self.q_table[n_ns][n_ew][n_light])
        td_target = reward + self.gamma * best_next
        self.q_table[ns][ew][light][action] += self.alpha * (
            td_target - self.q_table[ns][ew][light][action]
        )
 
# Train the agent
env = TrafficEnv()
agent = TrafficAgent()
episodes = 1000
reward_log = []
 
for ep in range(episodes):
    state = env.reset()
    ep_reward = 0
 
    for _ in range(50):  # simulate 50 time steps
        action = agent.choose_action(state)
        next_state, reward, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        ep_reward += reward
 
    reward_log.append(ep_reward)
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Total Reward: {ep_reward:.2f}")
 
# Plot performance
plt.plot(np.convolve(reward_log, np.ones(10)/10, mode='valid'))
plt.title("RL for Traffic Light Control – Queue Reduction")
plt.xlabel("Episode")
plt.ylabel("Total Reward (Higher = Shorter Queues)")
plt.grid(True)
plt.show()