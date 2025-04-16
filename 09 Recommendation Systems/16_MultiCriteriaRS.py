"""
Project 336. Multi-criteria recommendation system
Description:
In multi-criteria recommendation systems, multiple criteria are considered simultaneously when recommending items. For example:

Price, quality, and brand when recommending products

Duration, genre, and rating when recommending movies

In this project, we’ll build a multi-criteria recommendation system using weighted scoring to account for different factors.

About:
✅ What It Does:
Simulates a multi-criteria evaluation for items (e.g., price, quality, brand)

Uses user weights to give different importance to each criterion

Recommends items based on a weighted sum of the criteria
"""
import numpy as np
import pandas as pd
 
# 1. Simulate user-item ratings and criteria (e.g., price, quality, brand)
items = ['Item1', 'Item2', 'Item3', 'Item4', 'Item5']
criteria = ['price', 'quality', 'brand']
 
# Example data: price, quality, and brand ratings for each item
item_features = {
    'Item1': {'price': 300, 'quality': 4, 'brand': 5},
    'Item2': {'price': 50, 'quality': 3, 'brand': 3},
    'Item3': {'price': 150, 'quality': 5, 'brand': 4},
    'Item4': {'price': 40, 'quality': 2, 'brand': 3},
    'Item5': {'price': 200, 'quality': 4, 'brand': 4}
}
 
# User's importance weights for each criterion (e.g., they care more about quality)
user_weights = {
    'price': 0.3,   # User cares 30% about price
    'quality': 0.5, # User cares 50% about quality
    'brand': 0.2    # User cares 20% about brand
}
 
# 2. Multi-criteria recommendation logic (weighted sum of features)
def multi_criteria_recommendation(item_features, user_weights, top_n=3):
    scores = {}
    
    for item, features in item_features.items():
        score = sum(features[criterion] * user_weights[criterion] for criterion in criteria)
        scores[item] = score
 
    # Sort items by score in descending order
    recommended_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [item for item, _ in recommended_items[:top_n]]
 
# 3. Recommend top N items based on user's preferences (multi-criteria)
recommended_items = multi_criteria_recommendation(item_features, user_weights)
print("Multi-Criteria Recommendations for the user:", recommended_items)