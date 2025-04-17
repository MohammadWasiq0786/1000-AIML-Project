"""
Project 838. Product Categorization System

A product categorization system automatically assigns products to predefined categories based on their titles or descriptions. This is crucial for organizing online catalogs, improving search, and enabling filters. In this project, we simulate product titles and use a text classification model (Naive Bayes) to predict categories.

This basic classification system uses bag-of-words features to learn which words are associated with each product category. It’s ideal for e-commerce platforms and can be extended using:

TF-IDF or BERT embeddings for better text representation

Multi-label classification for products with overlapping categories

Integration with product metadata (e.g., brand, price, tags)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
 
# Simulated dataset of product titles and categories
data = {
    'Title': [
        'Wireless Bluetooth Headphones',
        'USB-C Fast Charging Cable',
        'Cotton Crew Neck T-Shirt',
        'Noise Cancelling Earbuds',
        'Men’s Slim Fit Jeans',
        'Portable Laptop Charger',
        'Silk Scarf for Women',
        '4K Ultra HD Smart TV',
        'Denim Jacket for Men',
        'Smartphone Screen Protector'
    ],
    'Category': [
        'Electronics', 'Electronics', 'Clothing', 'Electronics', 'Clothing',
        'Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics'
    ]
}
 
df = pd.DataFrame(data)
 
# Features (text) and target (category)
X = df['Title']
y = df['Category']
 
# Convert text to bag-of-words features
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)
 
# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)
 
# Predict on test data
y_pred = model.predict(X_test)
 
# Evaluate the classifier
print("Product Categorization Performance:")
print(classification_report(y_test, y_pred))
 
# Example: categorize a new product
new_product = ["Stylish Leather Wallet for Men"]
new_vector = vectorizer.transform(new_product)
predicted_category = model.predict(new_vector)[0]
print(f"\nPredicted Category for '{new_product[0]}': {predicted_category}")