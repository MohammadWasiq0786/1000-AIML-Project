"""
Project 327. Context-aware recommendation system
Description:
A context-aware recommendation system integrates additional contextual information (e.g., time of day, user location, weather, device type) to personalize recommendations. This is particularly useful when recommendations depend on the situation or environment.

In this project, we’ll implement a context-aware recommendation system where we consider the time of day to recommend items (e.g., recommending movies or products based on the user's current time).

About:
✅ What It Does:
Simulates session data with time of day context (morning, afternoon, evening)

Recommends items based on time of day, making recommendations contextually aware

Outputs item recommendations tailored to the user's context (e.g., time of day)
"""

import numpy as np
import pandas as pd
from datetime import datetime
 
# 1. Simulate session data with time of day and items
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
time_of_day = ['Morning', 'Afternoon', 'Evening']
 
data = []
for session in sessions:
    # Randomly pick an item and a time of day
    item = np.random.choice(items)
    time = np.random.choice(time_of_day)
    data.append([session, item, time, str(datetime.now())])
 
# Create DataFrame
df = pd.DataFrame(data, columns=['Session', 'Item', 'TimeOfDay', 'Timestamp'])
print("Session Data:\n", df)
 
# 2. Context-Aware Recommendation based on Time of Day
def recommend_based_on_context(session, time_of_day, df, top_n=1):
    # Filter data by session and time of day
    session_data = df[(df['Session'] == session) & (df['TimeOfDay'] == time_of_day)]
    
    # Recommend items based on time of day
    if time_of_day == 'Morning':
        recommendations = ['Item1', 'Item2']
    elif time_of_day == 'Afternoon':
        recommendations = ['Item3', 'Item4']
    else:
        recommendations = ['Item5', 'Item1']
    
    return recommendations[:top_n]
 
# 3. Recommend items for Session1 in the Afternoon
session = 'Session1'
time_of_day = 'Afternoon'
recommended_items = recommend_based_on_context(session, time_of_day, df)
print(f"Recommended items for {session} during {time_of_day}: {recommended_items}")