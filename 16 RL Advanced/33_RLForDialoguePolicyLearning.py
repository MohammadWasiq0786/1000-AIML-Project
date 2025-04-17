"""
Project 633: RL for Dialogue Policy Learning
Description:
Reinforcement learning (RL) for dialogue policy learning focuses on training agents to engage in conversations with users by learning optimal policies based on interaction rewards. The goal is to generate relevant, coherent, and helpful responses in dialogue systems. RL is particularly useful for dialogue policy learning because it allows the agent to improve its behavior over time by receiving feedback from users or simulated environments. In this project, we will apply RL to train a dialogue agent that learns how to interact with users to achieve predefined goals.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define a simple RNN-based model for dialogue response generation
class DialoguePolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DialoguePolicyNetwork, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Output: action (response) probabilities
 
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # Output action probabilities based on the last hidden state
 
# 2. Define the RL agent for dialogue policy learning using policy gradient
class RLDialogueAgent:
    def __init__(self, model, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.criterion = nn.CrossEntropyLoss()
 
    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(state))  # Random action (exploration)
        else:
            action_probs = self.model(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(action_probs).item()  # Select action with the highest probability
 
    def update(self, state, action, reward, next_state, done):
        # Calculate loss (policy gradient)
        action_probs = self.model(torch.tensor(state, dtype=torch.float32))
        log_prob = torch.log(torch.softmax(action_probs, dim=-1)[0, action])
        loss = -log_prob * reward  # Policy gradient: maximize reward
 
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        # Decay epsilon (exploration rate)
        if done:
            self.epsilon *= self.epsilon_decay
 
        return loss.item()
 
# 3. Define a simple reward function for evaluating generated responses
def simple_dialogue_reward(response):
    # Simple reward based on response length (longer and more relevant is better)
    return len(response)
 
# 4. Initialize the environment and RL agent for dialogue policy learning
input_size = 256  # Example size for word embeddings (dummy values)
output_size = 256  # Size of vocabulary (dummy values)
model = DialoguePolicyNetwork(input_size, output_size)
agent = RLDialogueAgent(model)
 
# 5. Train the agent using RL for dialogue policy learning
num_episodes = 1000
for episode in range(num_episodes):
    state = np.random.rand(1, 10, input_size)  # Random state (e.g., initial dialogue context)
    done = False
    total_reward = 0
    dialogue_history = []
    
    while not done:
        action = agent.select_action(state)
        next_state = np.random.rand(1, 10, input_size)  # Dummy next state (next response)
        reward = simple_dialogue_reward(dialogue_history)  # Compute reward (based on dialogue length)
        dialogue_history.append(action)  # Store generated response (for simplicity, use action as word)
 
        # Update the agent using the reward
        loss = agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
 
        if len(dialogue_history) > 20:  # Stop after a certain number of responses
            done = True
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss:.4f}")
 
# 6. Evaluate the agent after training (no exploration, only exploitation)
state = np.random.rand(1, 10, input_size)  # Initial state (dummy input)
done = False
dialogue_history = []
 
while not done:
    action = agent.select_action(state)
    next_state = np.random.rand(1, 10, input_size)
    reward = simple_dialogue_reward(dialogue_history)
    dialogue_history.append(action)
 
    if len(dialogue_history) > 20:
        done = True
 
print(f"Generated dialogue (length-based reward): {dialogue_history}")