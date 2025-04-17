"""
Project 617: Reinforcement Learning from Human Feedback
Description:
Reinforcement Learning from Human Feedback (RLHF) is a method where the agent learns not only from interactions with the environment but also from feedback provided by human evaluators. This feedback can guide the agent in achieving desired behavior, especially in tasks where predefined rewards are difficult to specify. In this project, we will implement a basic RLHF setup where the agent receives human feedback in the form of ratings or preferences to adjust its learning process.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: action probabilities
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Softmax for action probabilities
 
# 2. Define the RLHF agent
class RLHFAgent:
    def __init__(self, model, learning_rate=0.001, human_feedback_weight=0.5):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.human_feedback_weight = human_feedback_weight  # How much weight to give human feedback
 
    def select_action(self, state):
        # Select an action based on the policy's probability distribution
        action_probs = self.model(torch.tensor(state, dtype=torch.float32))
        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
        return action
 
    def update(self, states, actions, rewards, human_feedback):
        # Update the model using both rewards and human feedback
        self.optimizer.zero_grad()
        
        # Combine the rewards and human feedback into a single signal
        total_feedback = rewards + self.human_feedback_weight * human_feedback
 
        # Compute loss based on combined feedback (cross-entropy)
        action_probs = self.model(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        loss = -torch.mean(log_probs * total_feedback)  # Maximize feedback
 
        # Backpropagation
        loss.backward()
        self.optimizer.step()
 
        return loss.item()
 
    def get_human_feedback(self, trajectory):
        # Simulate human feedback (in real-life scenarios, this could be feedback from a user)
        return np.random.uniform(0, 1, size=len(trajectory))  # Random feedback for illustration
 
# 3. Initialize the environment and RLHF agent
env = gym.make('CartPole-v1')
model = PolicyNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = RLHFAgent(model)
 
# 4. Train the agent using RLHF (Human feedback incorporated)
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    states = []
    actions = []
    rewards = []
    trajectories = []
 
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
 
        # Collect trajectory (state-action-reward) for feedback
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        trajectories.append((state, action, reward))
 
        state = next_state
        total_reward += reward
 
    # Simulate human feedback for the trajectory
    human_feedback = agent.get_human_feedback(trajectories)
 
    # Update the model using RLHF
    loss = agent.update(torch.tensor(states, dtype=torch.float32), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32), human_feedback)
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss:.4f}")
 
# 5. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after RLHF training: {total_reward}")