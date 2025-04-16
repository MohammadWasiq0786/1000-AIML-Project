"""
Project 271. RL for healthcare decision making
Description:
In healthcare, RL can assist in sequential treatment planning, resource allocation, and personalized medicine. An RL agent can learn from historical patient data or simulations to recommend optimal actions — like when to administer medication, adjust therapy, or discharge a patient — based on patient states.

In this project, we simulate a treatment environment where the patient’s health evolves over time, and the agent learns a policy to maximize long-term health outcomes.

About:
✅ What It Does:
Simulates treatment decisions with evolving health state dynamics.

Balances treatment benefits against side effects or overtreatment penalties.

Learns to reach and maintain optimal health via reward maximization.

Foundation for clinical decision support systems, especially in chronic care or ICU management.
"""


import numpy as np
import matplotlib.pyplot as plt
import random
 
# Simulated healthcare environment
class HealthEnv:
    def __init__(self):
        self.states = [0, 1, 2, 3, 4]  # 0 = critical, 4 = healthy
        self.actions = [0, 1]  # 0 = no treatment, 1 = treat
        self.reset()
 
    def reset(self):
        self.state = random.randint(0, 2)  # start sick
        return self.state
 
    def step(self, action):
        # Transition dynamics
        if action == 1:  # treatment
            if self.state < 4:
                self.state += np.random.choice([1, 0], p=[0.8, 0.2])
        else:  # no treatment
            if self.state > 0:
                self.state -= np.random.choice([1, 0], p=[0.5, 0.5])
 
        # Reward: more for healthy, penalty for overtreating
        reward = self.state - (2 if action == 1 else 0)
        done = self.state == 4 or self.state == 0
        return self.state, reward, done
 
# Q-learning agent
class HealthAgent:
    def __init__(self, n_states, n_actions):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
 
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        return np.argmax(self.q_table[state])
 
    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])
 
# Setup
env = HealthEnv()
agent = HealthAgent(n_states=5, n_actions=2)
episodes = 1000
reward_log = []
 
for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
 
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
 
    reward_log.append(total_reward)
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Total Reward: {total_reward:.2f}")
 
# Plot learning curve
plt.plot(np.convolve(reward_log, np.ones(10)/10, mode='valid'))
plt.title("RL for Healthcare Decision Making")
plt.xlabel("Episode")
plt.ylabel("Total Reward (Higher = Better Health)")
plt.grid(True)
plt.show()