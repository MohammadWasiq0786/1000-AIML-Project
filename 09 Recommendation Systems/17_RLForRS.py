"""
Project 337. Reinforcement learning for recommendations
Description:
Reinforcement learning (RL) can be used to continuously improve recommendations based on user interactions. The system treats each recommendation as an action, and the user’s feedback (e.g., click, purchase) as the reward. The goal is to maximize long-term user engagement.

In this project, we’ll implement a simple Q-learning-based recommendation system, where the system learns to recommend items based on user interactions.

About:
✅ What It Does:
Simulates a user-item ratings matrix with binary feedback (1 = liked, 0 = not liked)

Uses Q-learning to update the Q-table based on user feedback

Recommends items based on the epsilon-greedy policy (exploration vs exploitation)

Updates the Q-table over time to improve recommendations
"""


import numpy as np
import pandas as pd
import random
 
# 1. Simulate a user-item ratings matrix (user interactions)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
 
# Ratings (1 - liked, 0 - not liked)
ratings = np.array([
    [1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [1, 0, 0, 1, 0]
])
 
df = pd.DataFrame(ratings, index=users, columns=items)
 
# 2. Q-learning for recommendation (initialize Q-table)
Q = np.zeros((len(users), len(items)))  # Q-table for storing state-action values
 
# Parameters
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 0.2 # Exploration factor (probability of random action)
 
# 3. Recommend an item for a user using epsilon-greedy policy
def recommend_item(user_idx):
    if random.uniform(0, 1) < epsilon:
        # Explore: recommend a random item
        return random.choice(range(len(items)))
    else:
        # Exploit: recommend the item with the highest Q-value
        return np.argmax(Q[user_idx])
 
# 4. Simulate the learning process (user interaction and Q-value updates)
for epoch in range(100):  # Simulate 100 rounds of interaction
    for user_idx in range(len(users)):
        item_idx = recommend_item(user_idx)  # Recommend an item for the user
        
        # Simulate user feedback (1 if liked, 0 if not liked)
        feedback = df.iloc[user_idx, item_idx]
        
        # Update Q-value using Q-learning formula
        reward = feedback  # Reward is the user’s feedback
        Q[user_idx, item_idx] += alpha * (reward + gamma * np.max(Q[user_idx]) - Q[user_idx, item_idx])
    
    # Optionally print Q-table after each epoch to observe learning progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Q-table")
        print(Q)
 
# 5. Recommend items for each user after learning
print("\nFinal Recommendations after Q-learning:")
for user_idx in range(len(users)):
    recommended_item_idx = np.argmax(Q[user_idx])
    print(f"Recommended item for {users[user_idx]}: {items[recommended_item_idx]}")