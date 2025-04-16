"""
Project 276. RL for dialogue systems
Description:
Dialogue systems (chatbots or virtual assistants) often use RL to improve response quality, user satisfaction, and long-term conversation goals. Traditional supervised training lacks feedback on multi-turn dialogue coherence or usefulness — RL helps by incorporating reward signals (like engagement, sentiment, or task success).

In this project, we simulate a simple task-oriented dialogue, where the agent must learn to ask questions that lead to task completion, and it is rewarded when the user goal is satisfied.

About:
✅ What It Does:
Simulates a simple task-oriented conversation with slot filling.

Agent learns to ask the right questions to complete the task.

Uses REINFORCE to update based on whether it completed the task.

Forms the basis of goal-driven dialogue systems in customer service, booking, and assistants.
"""


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
 
# Define a toy user goal: wants pizza and delivery
user_goal = {"food": "pizza", "service": "delivery"}
slots = list(user_goal.keys())
values = list(user_goal.values())
 
# Agent possible utterances (actions)
utterances = [
    "What food do you want?",
    "Dine in or delivery?",
    "Thanks for your order.",
    "Do you want sushi?",
    "Pickup or delivery?",
    "Goodbye."
]
action_size = len(utterances)
 
# Policy network (maps state to action distribution)
class DialoguePolicy(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
 
    def forward(self, state):
        return self.net(state)
 
# Environment simulation: respond with slot fill status
class DialogueEnv:
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.state = np.zeros(2)  # [food_filled, service_filled]
        self.done = False
        return self.state.copy()
 
    def step(self, action_idx):
        text = utterances[action_idx]
        reward = 0
 
        if "food" in text and self.state[0] == 0:
            self.state[0] = 1
        elif "delivery" in text or "pickup" in text:
            self.state[1] = 1
 
        if all(self.state == 1) and not self.done:
            reward = 1  # task completed
            self.done = True
 
        return self.state.copy(), reward, self.done
 
# Setup
env = DialogueEnv()
policy = DialoguePolicy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)
episodes = 1000
reward_history = []
 
for ep in range(episodes):
    state = env.reset()
    log_probs = []
    total_reward = 0
 
    for _ in range(5):  # max 5 dialogue turns
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits = policy(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
 
        next_state, reward, done = env.step(action.item())
        state = next_state
        total_reward += reward
        if done:
            break
 
    # REINFORCE loss
    loss = -torch.stack(log_probs).sum() * total_reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    reward_history.append(total_reward)
 
    if (ep + 1) % 100 == 0:
        avg_reward = sum(reward_history[-100:]) / 100
        print(f"Episode {ep+1}, Avg Task Completion Rate: {avg_reward:.2f}")
 
# Plot learning progress
plt.plot(np.convolve(reward_history, np.ones(10)/10, mode='valid'))
plt.title("RL for Dialogue System – Task Completion Rate")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.grid(True)
plt.show()