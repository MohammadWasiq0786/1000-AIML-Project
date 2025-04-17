"""
Project 632: RL for Natural Language Generation
Description:
Reinforcement learning (RL) for natural language generation (NLG) focuses on using RL to improve the quality of generated text by optimizing for long-term rewards rather than relying solely on supervised learning. RL allows models to be trained to generate text that aligns with specific goals, such as fluency, coherence, or relevance. In this project, we will use policy gradient methods to train a model to generate meaningful sentences based on a given reward function.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
 
# 1. Define a simple RNN-based generator model for text generation
class TextGenerationModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(TextGenerationModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Output: next word probability
 
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # Predict the next word based on the last hidden state
 
# 2. Define the RL agent for natural language generation using policy gradients
class RLTextGenerationAgent:
    def __init__(self, model, learning_rate=0.001, gamma=0.99):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss()
 
    def select_action(self, state):
        # Softmax to select next word based on the current state
        action_probs = self.model(torch.tensor(state, dtype=torch.float32))
        action = torch.multinomial(torch.softmax(action_probs, dim=-1), 1).item()
        return action
 
    def update(self, state, action, reward, next_state, done):
        # Calculate loss (policy gradient)
        action_probs = self.model(torch.tensor(state, dtype=torch.float32))
        log_prob = torch.log(torch.softmax(action_probs, dim=-1)[0, action])
        loss = -log_prob * reward  # Policy gradient: maximize reward
 
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
        return loss.item()
 
# 3. Define a simple reward function for evaluating generated text
def simple_reward_function(generated_text):
    # Simple reward based on length of the generated text (longer is better)
    return len(generated_text)
 
# 4. Initialize the environment and RL agent for natural language generation
input_size = 256  # Example size for word embeddings (dummy values)
output_size = 256  # Size of vocabulary (dummy values)
model = TextGenerationModel(input_size, output_size)
agent = RLTextGenerationAgent(model)
 
# 5. Train the agent using RL for natural language generation
num_episodes = 1000
for episode in range(num_episodes):
    state = np.random.rand(1, 10, input_size)  # Random state (e.g., a dummy initial sentence)
    done = False
    total_reward = 0
    generated_text = []
    
    while not done:
        action = agent.select_action(state)
        next_state = np.random.rand(1, 10, input_size)  # Dummy next state (generated next word)
        reward = simple_reward_function(generated_text)  # Compute reward (based on text length)
        generated_text.append(action)  # Store generated word (for simplicity, use action as word)
 
        # Update the agent using the reward
        loss = agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
 
        if len(generated_text) > 20:  # Stop when the text length reaches a certain threshold
            done = True
 
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss:.4f}")
 
# 6. Evaluate the agent after training (no exploration, only exploitation)
state = np.random.rand(1, 10, input_size)  # Initial state (dummy input)
done = False
generated_text = []
 
while not done:
    action = agent.select_action(state)
    next_state = np.random.rand(1, 10, input_size)
    reward = simple_reward_function(generated_text)
    generated_text.append(action)
 
    if len(generated_text) > 20:
        done = True
 
print(f"Generated text (length-based reward): {generated_text}")