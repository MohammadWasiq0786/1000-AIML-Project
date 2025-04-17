"""
Project 615: Evolutionary Reinforcement Learning
Description:
Evolutionary reinforcement learning (ERL) is inspired by biological evolution, where agents evolve through mechanisms such as mutation, crossover, and selection. In this project, we will implement evolutionary strategies for reinforcement learning, using an evolutionary algorithm to evolve policies that maximize rewards. This approach can be useful when dealing with complex environments where traditional RL methods might struggle.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define the policy network (a simple neural network for evolutionary strategies)
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
 
# 2. Define the Evolutionary Reinforcement Learning agent using Evolution Strategies
class EvolutionaryRLAgent:
    def __init__(self, model, population_size=20, mutation_rate=0.1, learning_rate=0.01):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
    def mutate(self, model):
        # Mutate the weights of the model by adding small random noise
        with torch.no_grad():
            for param in model.parameters():
                param.add_(self.mutation_rate * torch.randn_like(param))
 
    def select_action(self, state):
        # Sample an action from the policy's probability distribution
        action_probs = self.model(torch.tensor(state, dtype=torch.float32))
        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
        return action
 
    def evaluate(self, env, num_episodes=10):
        # Evaluate the model by running episodes and calculating the total reward
        total_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                total_reward += reward
                state = next_state
            total_rewards.append(total_reward)
        return np.mean(total_rewards)
 
    def train(self, env, num_generations=100):
        for generation in range(num_generations):
            population_rewards = []
            population_models = []
            
            # Generate population and evaluate each model
            for _ in range(self.population_size):
                clone_model = PolicyNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
                clone_model.load_state_dict(self.model.state_dict())  # Copy the model
                self.mutate(clone_model)  # Mutate the model
                
                # Evaluate the model's performance on the environment
                reward = self.evaluate(env)
                population_rewards.append(reward)
                population_models.append(clone_model)
 
            # Select the best models based on their performance
            best_models = np.argsort(population_rewards)[-int(self.population_size / 2):]  # Top 50% models
 
            # Update the model with the best models from the population
            self.model.load_state_dict(population_models[best_models[0]].state_dict())
            print(f"Generation {generation + 1}, Best Reward: {population_rewards[best_models[0]]}")
 
# 3. Initialize the environment and agent
env = gym.make('CartPole-v1')
model = PolicyNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = EvolutionaryRLAgent(model)
 
# 4. Train the agent using evolutionary reinforcement learning (evolution strategies)
agent.train(env)