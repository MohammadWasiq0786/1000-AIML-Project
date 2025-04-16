"""
Project 330. Group recommendation system
Description:
A group recommendation system provides recommendations for a group of users rather than individuals. The challenge lies in:

Aggregating preferences from multiple users

Dealing with conflicting tastes in the group

In this project, we’ll simulate a group of users selecting movies based on their collective preferences using a simple group aggregation method (e.g., majority voting or averaging ratings).

About:
✅ What It Does:
Simulates ratings from multiple users for different movies

Computes the average rating for each movie to generate group recommendations

Recommends the top N movies based on collective preferences
"""

import numpy as np
import pandas as pd
 
# 1. Simulate group of users with item ratings (e.g., movies)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5']
ratings = np.array([
    [5, 3, 0, 0, 4],
    [4, 0, 0, 3, 2],
    [3, 2, 0, 5, 4],
    [0, 0, 5, 4, 3],
    [2, 5, 0, 1, 0]
])
 
df = pd.DataFrame(ratings, index=users, columns=items)
 
# 2. Define a function to compute group preferences (average ratings)
def group_recommendations(df, top_n=3):
    # Calculate the average rating for each item across all users
    avg_ratings = df.mean(axis=0)
 
    # Sort items based on the average rating
    recommended_items = avg_ratings.sort_values(ascending=False)
 
    return recommended_items.head(top_n)
 
# 3. Recommend top N movies for the group
recommended_movies = group_recommendations(df)
print(f"Group Recommendations (Top 3 Movies):\n{recommended_movies}")