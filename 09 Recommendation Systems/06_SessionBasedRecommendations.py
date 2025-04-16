"""
Project 326. Session-based recommendations
Description:
Session-based recommendation systems focus on recommending items based on the current session or recent behavior rather than historical data from the user’s past sessions. These systems are essential for platforms where users don’t have much past data, such as e-commerce or video streaming.

In this project, we’ll implement a simple session-based recommendation model using item sequences and last-click behavior to suggest items.

About:
✅ What It Does:
Simulates a session-based data with multiple items clicked in each session

Recommends the next item based on the last clicked item in the session

Outputs a simple recommendation for the next item in the session (excluding items already clicked)
"""

import numpy as np
import pandas as pd
import random
 
# 1. Simulate session data (user clicks per session)
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
data = []
 
for session in sessions:
    # Randomly pick 3 items per session
    clicked_items = random.sample(items, 3)
    for item in clicked_items:
        data.append([session, item])
 
# Create DataFrame
df = pd.DataFrame(data, columns=['Session', 'Item'])
 
# 2. Implement a simple session-based recommendation system
def recommend_next_item(session, df, top_n=1):
    # Get last clicked item for the session
    last_clicked_item = df[df['Session'] == session].iloc[-1]['Item']
    
    # Find other items clicked in the same session
    other_items = df[df['Session'] == session]['Item'].values
    recommendations = [item for item in items if item not in other_items]
    
    return recommendations[:top_n]
 
# 3. Recommend next item for Session1
session = 'Session1'
recommended_items = recommend_next_item(session, df)
print(f"Recommended next item for {session}: {recommended_items[0]}")