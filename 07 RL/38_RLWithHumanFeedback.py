"""
Project 278. RL with human feedback
Description:
RLHF is the backbone of modern alignment in language models like ChatGPT. It uses human preferences (or simulated ones) to guide training. The core idea: instead of explicit rewards, the model learns from rankings or comparisons of different outputs — "Which response is better?" — and adjusts behavior accordingly.

In this project, we simulate RLHF by:

Generating multiple responses for prompts

Ranking them using a reward model (simulated human preferences)

Updating the policy using policy gradients (REINFORCE)

About:
✅ What It Does:
Mimics RLHF using preference-based rewards.

Uses a policy network to generate responses to prompts.

Updates the policy using REINFORCE, guided by simulated human approval.

Scales to real-world use with ranking models and pairwise comparisons.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
 
# Simple simulated prompt-response environment
prompts = ["Tell me a joke", "Say something wise", "Motivate me"]
 
# Vocabulary and generator output options
responses = {
    "Tell me a joke": ["Why did the chicken cross the road?", "I don’t know any jokes.", "Knock knock!"],
    "Say something wise": ["Knowledge is power.", "Life is short.", "YOLO."],
    "Motivate me": ["You got this!", "Give up now.", "Keep pushing forward!"]
}
 
# Simulated human feedback (preference ranking)
def simulated_human_preference(prompt, response):
    # "Good" responses ranked higher
    good = {
        "Tell me a joke": "Knock knock!",
        "Say something wise": "Knowledge is power.",
        "Motivate me": "Keep pushing forward!"
    }
    return 1.0 if response == good[prompt] else 0.0
 
# Policy network: generates response index given prompt
class PolicyNet(nn.Module):
    def __init__(self, n_prompts, n_responses):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_prompts, 32), nn.ReLU(),
            nn.Linear(32, n_responses)
        )
 
    def forward(self, prompt_idx):
        x = torch.eye(len(prompts))[prompt_idx]  # one-hot encoding
        return self.model(x)
 
# Setup
n_prompts = len(prompts)
n_responses = 3
policy = PolicyNet(n_prompts, n_responses)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
reward_log = []
 
# Training loop
for ep in range(500):
    prompt_idx = random.randint(0, n_prompts - 1)
    prompt = prompts[prompt_idx]
 
    logits = policy(prompt_idx)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    response_idx = dist.sample()
    response = responses[prompt][response_idx.item()]
 
    # Simulated human reward
    reward = simulated_human_preference(prompt, response)
 
    # Policy gradient loss
    loss = -dist.log_prob(response_idx) * reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    reward_log.append(reward)
    if (ep + 1) % 50 == 0:
        avg = sum(reward_log[-50:]) / 50
        print(f"Episode {ep+1}, Prompt: '{prompt}', Response: '{response}', Avg Reward: {avg:.2f}")
 
# Plotting reward trend
plt.plot(np.convolve(reward_log, np.ones(10)/10, mode='valid'))
plt.title("RLHF Simulation – Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Reward (Preference Score)")
plt.grid(True)
plt.show()