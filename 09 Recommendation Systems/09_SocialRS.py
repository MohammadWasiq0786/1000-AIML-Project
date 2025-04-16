"""
Project 329. Social recommendation system
Description:
A social recommendation system uses a user’s social network (e.g., friends, followers) to make personalized recommendations. The idea is that people in similar social circles are likely to have similar preferences. This can include:

Friends' recommendations

Social influence (what peers like)

Shared interests or activities

In this project, we’ll simulate a social recommendation system where users are recommended items based on their friends' preferences.

About:
✅ What It Does:
Simulates a social network where users have friends with shared preferences

Recommends items based on the average ratings from a user’s friends

Filters out items the user has already rated, and ranks recommendations
"""

import numpy as np
import pandas as pd
 
# 1. Simulate a social network (user-item interactions and friendships)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
ratings = np.array([
    [5, 4, 0, 0, 3],
    [4, 0, 0, 2, 1],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 4],
    [2, 3, 0, 1, 0]
])
 
df = pd.DataFrame(ratings, index=users, columns=items)
 
# 2. Simulate friendships (social network)
friends = {
    'User1': ['User2', 'User3'],
    'User2': ['User1', 'User4'],
    'User3': ['User1', 'User5'],
    'User4': ['User2'],
    'User5': ['User3']
}
 
# 3. Social recommendation logic: Recommend items that friends liked
def social_recommendations(user, df, friends, top_n=2):
    # Get items rated by friends
    friends_ratings = []
    for friend in friends[user]:
        friend_ratings = df.loc[friend]
        friends_ratings.append(friend_ratings)
 
    # Combine ratings from friends
    combined_ratings = np.mean(friends_ratings, axis=0)
 
    # Recommend items that the user hasn't rated yet (top N by rating)
    unrated_items = df.loc[user].isna()
    recommended_items = [(item, combined_ratings[i]) for i, item in enumerate(df.columns) if unrated_items[i]]
    recommended_items = sorted(recommended_items, key=lambda x: x[1], reverse=True)
 
    return [item for item, _ in recommended_items[:top_n]]
 
# 4. Recommend items for User1 based on friends' preferences
user = 'User1'
recommended_items = social_recommendations(user, df, friends)
print(f"Social Recommendations for {user}: {recommended_items}")