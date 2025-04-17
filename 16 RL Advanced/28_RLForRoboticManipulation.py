"""
Project 628: RL for Robotic Manipulation
Description:
Reinforcement learning (RL) for robotic manipulation focuses on training robots to perform tasks that require interacting with the physical world, such as grasping objects, assembling parts, or moving items. This project will apply RL to robotic manipulation tasks, where the agent learns to control a robot's movements to optimize performance in tasks like object picking, placing, or other manipulation challenges.
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define the neural network model for the actor and critic (same as before)
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: continuous action space
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output actions within the continuous range (-1, 1)
 
class Critic(nn.Module):
    def __init__(self, input_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size + action_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Output: Q-value for a given state-action pair
 
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
 
# 2. Define the DDPG agent (same as before)
class DDPGAgent:
    def __init__(self, actor, critic, actor_lr=0.001, critic_lr=0.002, gamma=0.99, tau=0.005):
        self.actor = actor
        self.critic = critic
        self.target_actor = Actor(actor.input_size, actor.output_size)
        self.target_critic = Critic(critic.input_size, critic.action_size)
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau  # Soft target updates
 
        # Initialize target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
 
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.actor(state).detach().numpy()[0]  # Continuous action
 
    def update(self, state, action, reward, next_state, done):
        # Compute target Q-value
        next_action = self.target_actor(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))
        target_q_value = reward + (1 - done) * self.gamma * self.target_critic(next_state, next_action).detach()
 
        # Critic loss
        current_q_value = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q_value, target_q_value)
 
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
 
        # Actor loss (using the critic's Q-value)
        actor_loss = -self.critic(state, self.actor(state)).mean()  # Maximize Q-value
 
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
 
        # Soft target updates
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
 
        return actor_loss.item(), critic_loss.item()
 
    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
 
# 3. Initialize the environment and DDPG agent (robotic manipulation task)
env = gym.make('FetchPickAndPlace-v1')  # Example environment for robotic manipulation
actor = Actor(input_size=env.observation_space.shape[0], output_size=env.action_space.shape[0])
critic = Critic(input_size=env.observation_space.shape[0], action_size=env.action_space.shape[0])
agent = DDPGAgent(actor, critic)
 
# 4. Train the agent using DDPG for robotic manipulation
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
 
        # Update the agent using DDPG
        actor_loss, critic_loss = agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
 
# 5. Evaluate the agent after training (no exploration, only exploitation)
state = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
 
print(f"Total reward after DDPG training for Robotic Manipulation: {total_reward}")