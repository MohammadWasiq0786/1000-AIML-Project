"""
Project 275. RL for text generation
Description:
In language models, Reinforcement Learning can fine-tune generated text based on custom reward functions — like human preferences, grammatical correctness, or content alignment. This is the foundation of RLHF (Reinforcement Learning with Human Feedback), used in models like ChatGPT.

In this project, we simulate RL-based text generation using a policy gradient (REINFORCE) method to train a simple sequence generator to produce text that ends in a specific target word.

About:
✅ What It Does:
Generates a sequence of words with a reward based on ending.

Uses policy gradient to learn to prefer desired outputs.

Foundation for RLHF, toxicity filtering, and alignment tuning.

Can be extended to work with full language models + custom reward models.
"""


import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
 
# Simple vocabulary and tokens
vocab = ['hello', 'world', 'i', 'love', 'you', 'bye']
word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}
vocab_size = len(vocab)
 
# Goal: generate a sequence ending with "you"
target_word = 'you'
target_ix = word_to_ix[target_word]
 
# Policy network (RNN-based)
class TextPolicy(nn.Module):
    def __init__(self, vocab_size, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
 
    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden
 
# Initialize
model = TextPolicy(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
max_len = 5
episodes = 1000
reward_history = []
 
for ep in range(episodes):
    state = torch.LongTensor([[random.randint(0, vocab_size - 1)]])
    hidden = None
    log_probs = []
    sequence = [state.item()]
 
    for t in range(max_len - 1):
        logits, hidden = model(state, hidden)
        probs = torch.softmax(logits[0, -1], dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
 
        sequence.append(action.item())
        state = action.unsqueeze(0).unsqueeze(0)
 
    # Reward: +1 if final word is 'you', else 0
    reward = 1.0 if sequence[-1] == target_ix else 0.0
 
    # Policy gradient loss
    loss = -torch.stack(log_probs).sum() * reward
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    reward_history.append(reward)
    if (ep + 1) % 100 == 0:
        avg_r = sum(reward_history[-100:]) / 100
        generated = ' '.join([ix_to_word[i] for i in sequence])
        print(f"Episode {ep+1}, Avg Reward: {avg_r:.2f}, Example: '{generated}'")
 
# Plot reward curve
plt.plot(np.convolve(reward_history, np.ones(10)/10, mode='valid'))
plt.title("RL for Text Generation (REINFORCE)")
plt.xlabel("Episode")
plt.ylabel("Success Rate (Ends with 'you')")
plt.grid(True)
plt.show()