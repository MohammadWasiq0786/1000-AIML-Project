"""
Project 335. Cold-start problem solutions
Description:
The cold-start problem occurs when there is insufficient data for making recommendations. It commonly happens in:

New users with no ratings (user-based cold start)

New items with no interactions (item-based cold start)

In this project, we’ll explore two techniques to mitigate the cold-start problem:

Content-based methods: Recommend items based on their features, especially useful for new items.

Hybrid approaches: Combine content-based recommendations and collaborative filtering for new users.

About:
✅ What It Does:
Content-based recommendations for new items based on item features (e.g., category, price)

Hybrid recommendations for new users using item features since they have no prior ratings

Provides solutions to the cold-start problem by recommending based on available information (features instead of ratings)
"""

import numpy as np
import pandas as pd
 
# 1. Simulate ratings for users and items (with cold-start problem)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
 
# Ratings matrix for users and items
ratings = np.array([
    [5, 0, 0, 0, 4],
    [4, 0, 0, 3, 2],
    [0, 3, 0, 5, 4],
    [2, 0, 5, 4, 0],
    [3, 3, 0, 1, 0]
])
 
df = pd.DataFrame(ratings, index=users, columns=items)
 
# 2. Simulate item features (for content-based recommendations)
item_features = {
    'Item1': {'category': 'Electronics', 'brand': 'BrandA', 'price': 300},
    'Item2': {'category': 'Clothing', 'brand': 'BrandB', 'price': 50},
    'Item3': {'category': 'Electronics', 'brand': 'BrandC', 'price': 150},
    'Item4': {'category': 'Clothing', 'brand': 'BrandA', 'price': 40},
    'Item5': {'category': 'Furniture', 'brand': 'BrandB', 'price': 200}
}
 
# 3. Cold-start solution for new items (using content-based filtering)
def recommend_new_item(item_features, top_n=2):
    # Recommend new items based on their features (content-based)
    sorted_items = sorted(item_features.items(), key=lambda x: x[1]['price'], reverse=True)  # Example: sorting by price
    return [item[0] for item in sorted_items[:top_n]]
 
# 4. Cold-start solution for new users (using hybrid approach)
def recommend_new_user(user_ratings, df, item_features, top_n=2):
    # Recommend based on item features for a new user (content-based approach)
    recommended_items = recommend_new_item(item_features, top_n)
    
    # Optionally, combine with collaborative filtering if some ratings exist (hybrid approach)
    return recommended_items
 
# 5. Recommend items for a new item (using content-based method)
recommended_for_new_item = recommend_new_item(item_features)
print("Recommendations for new item (based on features):", recommended_for_new_item)
 
# 6. Recommend items for a new user (using hybrid approach)
user_ratings = []  # New user has no ratings
recommended_for_new_user = recommend_new_user(user_ratings, df, item_features)
print("Recommendations for new user (based on features):", recommended_for_new_user)