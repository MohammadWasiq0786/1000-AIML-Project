"""
Project 348. Location-aware recommendations
Description:
Location-aware recommendation systems leverage a user’s location to provide personalized recommendations based on geographic context. This is ideal for scenarios such as:

Recommending local businesses or restaurants

Location-based services (e.g., events, attractions)

Personalized offers based on proximity (e.g., discounts at nearby stores)

In this project, we’ll build a location-aware recommendation system that uses geographical coordinates (latitude and longitude) to recommend items or services in the user’s vicinity.

About:
✅ What It Does:
Simulates a user-item ratings matrix with location-based items (e.g., restaurants, stores)

Uses geodesic distance (latitude and longitude) to calculate the distance from the user’s location to each item

Recommends items based on proximity and user ratings, prioritizing closer and highly-rated items
"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic
 
# 1. Simulate user-item ratings matrix (location-based items)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
items = ['Restaurant1', 'Restaurant2', 'Restaurant3', 'Store1', 'Store2']
ratings = np.array([
    [5, 4, 0, 0, 3],
    [4, 0, 0, 3, 2],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 4],
    [2, 3, 0, 1, 0]
])
 
# Define locations (latitude, longitude) for each item (e.g., businesses, restaurants)
item_locations = {
    'Restaurant1': (40.748817, -73.985428),  # New York
    'Restaurant2': (34.052235, -118.243683),  # Los Angeles
    'Restaurant3': (51.507351, -0.127758),   # London
    'Store1': (48.856613, 2.352222),         # Paris
    'Store2': (35.6762, 139.6503)            # Tokyo
}
 
# User's current location (latitude, longitude)
user_location = (40.712776, -74.005974)  # New York
 
# 2. Function to calculate distance from the user's location to each item
def get_distance(user_location, item_location):
    return geodesic(user_location, item_location).km  # Distance in kilometers
 
# 3. Location-aware recommendation logic
def location_aware_recommendation(user_idx, df, item_locations, user_location, top_n=3):
    # Filter the ratings of the user
    user_ratings = df[df['user'] == users[user_idx]]
    
    # Create a dictionary to store distance and rating for each item
    item_distances = {}
    for item in items:
        # Get the item's location
        item_location = item_locations.get(item, None)
        if item_location:
            # Calculate the distance from the user's location to the item
            distance = get_distance(user_location, item_location)
            rating = df.iloc[user_idx, items.index(item)]  # User's rating for the item
            item_distances[item] = (distance, rating)
    
    # Sort items by proximity (distance) and rating
    sorted_items = sorted(item_distances.items(), key=lambda x: (x[1][0], -x[1][1]))  # Sort by distance, then by rating
    
    # Recommend the top N items
    recommended_items = [item for item, _ in sorted_items[:top_n]]
    
    return recommended_items
 
# 4. Recommend items for User1 (index 0) based on their location and ratings
user_idx = 0  # User1
recommended_items = location_aware_recommendation(user_idx, df, item_locations, user_location)
print(f"Location-Aware Recommendations for User1: {recommended_items}")