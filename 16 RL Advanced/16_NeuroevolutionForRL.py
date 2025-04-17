"""
Project 616: Neuroevolution for Reinforcement Learning
Description:
Neuroevolution refers to the process of evolving neural network architectures using evolutionary algorithms, where the parameters or structures of the network are optimized over generations. In the context of reinforcement learning (RL), neuroevolution can be used to evolve both the policy network and the structure of the neural network that governs the agent's actions. This project will implement neuroevolution for reinforcement learning by evolving the network weights of a reinforcement learning agent.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
# 1. Define the neural network model (for neuroevolution)
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Softmax for action probabilities
 
# 2. Define the agent using genetic algorithms (neuroevolution)
class NeuroevolutionAgent:
    def __init__(self, model, population_size=20, mutation_rate=0.1, learning_rate=0.01):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
    def mutate(self, model):
        # Mutate the neural network by adding small random noise to its weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(self.mutation_rate * torch.randn_like(param))
 
    def select_action(self, state):
        # Select an action based on the policy's probability distribution
        action_probs = self.model(torch.tensor(state, dtype=torch.float32))
        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
        return action
 
    def evaluate(self, env, num_episodes=10):
        # Evaluate the model by running it in the environment and calculating the average reward
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
            
            # Generate the population and evaluate each model
            for _ in range(self.population_size):
                clone_model = NeuralNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
                clone_model.load_state_dict(self.model.state_dict())  # Copy the model
                self.mutate(clone_model)  # Apply mutation (neuroevolution)
                
                # Evaluate the model's performance
                reward = self.evaluate(env)
                population_rewards.append(reward)
                population_models.append(clone_model)
 
            # Select the top-performing models based on their reward
            best_models = np.argsort(population_rewards)[-int(self.population_size / 2):]  # Top 50% models
 
            # Update the model with the best models from the population
            self.model.load_state_dict(population_models[best_models[0]].state_dict())
            print(f"Generation {generation + 1}, Best Reward: {population_rewards[best_models[0]]}")
 
# 3. Initialize the environment and agent
env = gym.make('CartPole-v1')
model = NeuralNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = NeuroevolutionAgent(model)
 
# 4. Train the agent using neuroevolution (genetic algorithm)
agent.train(env)