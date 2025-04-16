"""
Project 332. Sequential recommendation system
Description:
Sequential recommendation systems consider the order of user interactions, such as:

Browsing history

Clickstream data

Time series of actions

These systems are ideal for scenarios like:

E-commerce (recommending products based on browsing sequence)

Video streaming (recommending the next episode or movie)

News articles (recommending articles based on reading history)

In this project, we’ll implement a simple sequence-based recommendation using a Markov chain to predict the next item based on the previous actions.

About:
✅ What It Does:
Simulates sequences of user-item interactions (like browsing history)

Builds a transition matrix based on item-to-item transitions

Uses the matrix to recommend the next item in the sequence for a user
"""

import numpy as np
import pandas as pd
 
# 1. Simulate user-item interactions in sequence
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
sequences = {
    'User1': ['Item1', 'Item2', 'Item3', 'Item4'],
    'User2': ['Item1', 'Item3', 'Item5'],
    'User3': ['Item2', 'Item4', 'Item5'],
    'User4': ['Item1', 'Item2', 'Item4'],
    'User5': ['Item3', 'Item5']
}
 
# 2. Build a transition matrix based on item sequences
transition_matrix = {item: {next_item: 0 for next_item in items} for item in items}
 
# Update transition counts based on user interactions
for user, seq in sequences.items():
    for i in range(len(seq) - 1):
        current_item = seq[i]
        next_item = seq[i+1]
        transition_matrix[current_item][next_item] += 1
 
# 3. Normalize the transition matrix to get probabilities
for current_item, next_items in transition_matrix.items():
    total_transitions = sum(next_items.values())
    if total_transitions > 0:
        for next_item in next_items:
            transition_matrix[current_item][next_item] /= total_transitions
 
# 4. Recommend next item based on sequence
def recommend_next_item(user, sequences, transition_matrix):
    last_item = sequences[user][-1]
    next_item_probabilities = transition_matrix[last_item]
    
    # Sort items by the highest probability of transition
    sorted_items = sorted(next_item_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Recommend the item with the highest transition probability
    recommended_item = sorted_items[0][0]
    return recommended_item
 
# 5. Recommend the next item for User1
user = 'User1'
recommended_item = recommend_next_item(user, sequences, transition_matrix)
print(f"Recommended next item for {user}: {recommended_item}")