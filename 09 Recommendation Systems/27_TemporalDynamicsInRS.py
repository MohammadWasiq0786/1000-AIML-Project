"""
Project 347. Temporal dynamics in recommendations
Description:
Temporal dynamics in recommendation systems refers to the ability to capture the time-sensitive nature of user preferences. This is important in scenarios where:

User preferences change over time (e.g., movies, music, or news)

Certain items may be seasonal or relevant at different times (e.g., holiday decorations, fashion trends)

In this project, we’ll build a time-aware recommendation system that accounts for the temporal evolution of user preferences.

About:
✅ What It Does:
Simulates a user-item ratings matrix with timestamps to represent time-based interactions

Recommends items based on the recency of user interactions, giving more weight to recent preferences

Sorts user ratings by time difference from the current date to provide recommendations that reflect temporal dynamics
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
 
# 1. Simulate user-item ratings matrix (with timestamps)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
 
# Create ratings with timestamps (1-5 scale) over a period of time
timestamps = [
    datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2023, 3, 1), datetime(2023, 4, 1), datetime(2023, 5, 1)
]
ratings = np.array([
    [5, 4, 0, 0, 3],
    [4, 0, 0, 3, 2],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 4],
    [2, 3, 0, 1, 0]
])
 
# Create a DataFrame with ratings and timestamps
data = []
for user_idx, user in enumerate(users):
    for item_idx, item in enumerate(items):
        if ratings[user_idx, item_idx] > 0:
            data.append([user, item, ratings[user_idx, item_idx], timestamps[item_idx]])
 
df = pd.DataFrame(data, columns=['user', 'item', 'rating', 'timestamp'])
 
# 2. Temporal recommendation system based on recent preferences
def temporal_recommendation(user_idx, df, top_n=3):
    # Filter the ratings of the user
    user_ratings = df[df['user'] == users[user_idx]]
    
    # Sort the items based on most recent ratings
    user_ratings['recency'] = (datetime.now() - user_ratings['timestamp']).dt.total_seconds()
    user_ratings_sorted = user_ratings.sort_values(by='recency', ascending=True)
    
    # Recommend the top N items based on recency of user interactions
    recommended_items = user_ratings_sorted.head(top_n)['item'].tolist()
    
    return recommended_items
 
# 3. Recommend items for User1 (index 0) based on recent interactions
user_idx = 0  # User1
recommended_items = temporal_recommendation(user_idx, df)
print(f"Temporal Recommendations for User1: {recommended_items}")