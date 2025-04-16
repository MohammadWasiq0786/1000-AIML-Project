"""
Project 349. Conversational recommendation system
Description:
A conversational recommendation system engages users in a chat-based interface to suggest items based on their preferences and queries. Instead of simply providing a list of items, this system:

Asks questions to understand user preferences

Provides dynamic suggestions based on real-time feedback

Offers a personalized experience based on user responses

In this project, we’ll create a simple conversational agent that recommends items based on the user’s preferences (e.g., product type, price range, etc.).

About:
✅ What It Does:
The conversational agent asks the user about their product category and price range preferences

Filters the products based on the user's responses (e.g., price and category)

Provides personalized recommendations based on the user’s answers
"""

import random
 
# 1. Simulate items and user preferences
items = ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch']
item_details = {
    'Laptop': {'category': 'Electronics', 'price': 1000, 'brand': 'BrandA'},
    'Smartphone': {'category': 'Electronics', 'price': 800, 'brand': 'BrandB'},
    'Headphones': {'category': 'Electronics', 'price': 150, 'brand': 'BrandC'},
    'Tablet': {'category': 'Electronics', 'price': 500, 'brand': 'BrandA'},
    'Smartwatch': {'category': 'Electronics', 'price': 200, 'brand': 'BrandB'}
}
 
# 2. Define the conversational agent
class ConversationalRecommender:
    def __init__(self, items, item_details):
        self.items = items
        self.item_details = item_details
    
    def ask_question(self, question):
        print(question)
        response = input("Your answer: ").strip().lower()
        return response
    
    def recommend(self):
        print("Hello! I'm here to help you find the best products.")
        
        # Ask about category preference
        category_response = self.ask_question("What category of product are you interested in? (e.g., Electronics)")
        
        # Ask about price range preference
        price_response = self.ask_question("What's your preferred price range? (e.g., low, medium, high)")
        
        # Filter items based on category and price
        recommended_items = []
        for item in self.items:
            details = self.item_details[item]
            
            # Filtering based on category
            if category_response in details['category'].lower():
                # Filtering based on price
                if price_response == 'low' and details['price'] < 300:
                    recommended_items.append(item)
                elif price_response == 'medium' and 300 <= details['price'] <= 800:
                    recommended_items.append(item)
                elif price_response == 'high' and details['price'] > 800:
                    recommended_items.append(item)
        
        if not recommended_items:
            print("Sorry, I couldn't find any products matching your preferences.")
        else:
            print(f"Based on your preferences, I recommend the following products: {', '.join(recommended_items)}")
 
# 3. Create and interact with the recommender
recommender = ConversationalRecommender(items, item_details)
recommender.recommend()