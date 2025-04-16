"""
Project 270. RL for energy management
Description:
In smart grids and buildings, energy management systems aim to balance power usage, storage, and generation (e.g., solar) to minimize cost and maximize efficiency. RL can help by adapting energy usage strategies based on electricity prices, battery levels, and forecasted demand.

In this project, we simulate a simple home energy system with a battery, energy demand, and varying prices. The RL agent learns to charge, discharge, or stay idle to optimize total cost over a day.

About:
✅ What It Does:
Simulates 24-hour energy scheduling with time-based pricing.

Learns when to charge (cheap) or discharge (expensive) to minimize cost.

Models demand, storage limits, and dynamic decisions.

Can be scaled to smart homes, microgrids, or data centers.
"""


import numpy as np
import matplotlib.pyplot as plt
import random
 
# Simulated energy environment
class EnergyEnv:
    def __init__(self):
        self.hours = 24
        self.battery_capacity = 10
        self.max_action = 1  # max charge/discharge per step
        self.reset()
 
    def reset(self):
        self.time = 0
        self.battery = 5  # start half full
        return (self.time, self.battery)
 
    def step(self, action):
        # action ∈ {-1, 0, 1}: discharge, idle, charge
        price = self.get_price(self.time)
        demand = self.get_demand(self.time)
 
        self.battery = np.clip(self.battery + action, 0, self.battery_capacity)
        supply = action if action > 0 else 0
        grid_use = max(demand - supply, 0)
        cost = grid_use * price
 
        self.time += 1
        done = self.time >= self.hours
        state = (self.time, self.battery)
        reward = -cost
        return state, reward, done
 
    def get_price(self, hour):
        # Simulated time-of-use pricing (e.g., peak in evening)
        return 0.1 + 0.3 * (hour >= 18)
 
    def get_demand(self, hour):
        return 1 + int(18 <= hour <= 22)  # peak demand 6–10 PM
 
# Q-learning agent for energy control
class EnergyAgent:
    def __init__(self, hours, battery_capacity):
        self.q_table = np.zeros((hours + 1, battery_capacity + 1, 3))  # actions: -1,0,1
        self.actions = [-1, 0, 1]
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2
 
    def choose_action(self, time, battery):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[time][battery])]
 
    def update(self, time, battery, action, reward, next_time, next_battery):
        a_idx = self.actions.index(action)
        next_q = max(self.q_table[next_time][next_battery])
        td_target = reward + self.gamma * next_q
        self.q_table[time][battery][a_idx] += self.alpha * (
            td_target - self.q_table[time][battery][a_idx]
        )
 
# Training
env = EnergyEnv()
agent = EnergyAgent(hours=env.hours, battery_capacity=env.battery_capacity)
 
episodes = 1000
total_rewards = []
 
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
 
    while not done:
        t, b = state
        action = agent.choose_action(t, b)
        next_state, reward, done = env.step(action)
        nt, nb = next_state
 
        agent.update(t, b, action, reward, nt, nb)
        state = next_state
        total_reward += reward
 
    total_rewards.append(total_reward)
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Total Reward (−Cost): {total_reward:.2f}")
 
# Plotting reward (−Cost)
plt.plot(np.convolve(total_rewards, np.ones(10)/10, mode='valid'))
plt.title("RL for Energy Management – Cost Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward (Higher = Lower Cost)")
plt.grid(True)
plt.show()