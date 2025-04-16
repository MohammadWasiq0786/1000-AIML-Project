"""
Project 328. Knowledge-based recommendation system
Description:
A knowledge-based recommendation system uses explicit knowledge (like user preferences, expert rules, or item features) to make recommendations, rather than relying on user behavior. This is ideal for domains where:

Explicit preferences matter (e.g., product features)

Items are complex, and users need guidance

In this project, we’ll implement a simple knowledge-based recommendation system for recommending laptops based on user preferences for features like battery life, price, and brand.

About:
✅ What It Does:
Simulates a knowledge-based system for recommending laptops

Filters laptops based on user preferences (brand, battery life, price)

Outputs laptop recommendations based on expert knowledge or predefined rules
"""

import numpy as np
import pandas as pd
 
# 1. Simulate a knowledge-based recommendation system for laptops
laptops = ['Laptop1', 'Laptop2', 'Laptop3', 'Laptop4', 'Laptop5']
features = {
    'Laptop1': {'brand': 'BrandA', 'battery_life': 10, 'price': 500},
    'Laptop2': {'brand': 'BrandB', 'battery_life': 8, 'price': 700},
    'Laptop3': {'brand': 'BrandC', 'battery_life': 12, 'price': 600},
    'Laptop4': {'brand': 'BrandA', 'battery_life': 6, 'price': 400},
    'Laptop5': {'brand': 'BrandB', 'battery_life': 9, 'price': 650},
}
 
# 2. User preferences (input)
user_preferences = {
    'brand': 'BrandA',
    'min_battery_life': 8,
    'max_price': 600
}
 
# 3. Knowledge-based recommendation logic (filter laptops based on preferences)
def recommend_laptops(user_preferences, features):
    recommended_laptops = []
    for laptop, specs in features.items():
        if (specs['brand'] == user_preferences['brand'] and
            specs['battery_life'] >= user_preferences['min_battery_life'] and
            specs['price'] <= user_preferences['max_price']):
            recommended_laptops.append(laptop)
    return recommended_laptops
 
# 4. Get recommendations
recommended_laptops = recommend_laptops(user_preferences, features)
print(f"Recommended Laptops based on your preferences: {recommended_laptops}")