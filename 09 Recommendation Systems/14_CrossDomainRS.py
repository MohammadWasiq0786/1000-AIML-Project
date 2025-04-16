"""
Project 334. Cross-domain recommendation system
Description:
A cross-domain recommendation system suggests items from one domain (e.g., movies) based on preferences in another domain (e.g., books). This is useful for:

Users with sparse data in one domain

Recommending complementary items from other domains

Expanding user interest by transferring knowledge between domains

In this project, we’ll build a cross-domain recommendation system where user preferences in books will help recommend movies.

About:
✅ What It Does:
Simulates user-item ratings for both books and movies

Recommends movies based on the user’s ratings in books by correlating preferences across domains

Provides cross-domain recommendations to expand user interests (e.g., from books to movies)
"""

import numpy as np
import pandas as pd
 
# 1. Simulate user-item ratings for books and movies (cross-domain)
users = ['User1', 'User2', 'User3', 'User4', 'User5']
books = ['Book1', 'Book2', 'Book3', 'Book4', 'Book5']
movies = ['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5']
 
# Ratings for books (1 to 5 scale)
book_ratings = np.array([
    [5, 0, 0, 0, 4],
    [4, 0, 0, 3, 2],
    [0, 3, 0, 5, 4],
    [2, 0, 5, 4, 0],
    [3, 3, 0, 1, 0]
])
 
# Ratings for movies (1 to 5 scale)
movie_ratings = np.array([
    [5, 4, 0, 0, 3],
    [4, 0, 0, 2, 1],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 4],
    [2, 3, 0, 1, 0]
])
 
# Create DataFrames for both domains
book_df = pd.DataFrame(book_ratings, index=users, columns=books)
movie_df = pd.DataFrame(movie_ratings, index=users, columns=movies)
 
# 2. Cross-domain recommendation logic
def cross_domain_recommendation(user_idx, book_df, movie_df, top_n=3):
    # Get user preferences from books
    book_ratings_user = book_df.iloc[user_idx]
    rated_books = book_ratings_user[book_ratings_user > 0].index  # Books rated by the user
 
    # Find items in the movie domain that correlate with the books the user liked
    # Simple approach: recommend movies based on the user's highest rated books
    recommended_movies = []
    for book in rated_books:
        book_rating = book_ratings_user[book]
        similar_movies = movie_df[book].sort_values(ascending=False).head(top_n)
        
        for movie in similar_movies.index:
            if movie not in recommended_movies:
                recommended_movies.append(movie)
    
    return recommended_movies[:top_n]
 
# 3. Recommend movies for User1 based on their book ratings
user_idx = 0  # User1
recommended_movies = cross_domain_recommendation(user_idx, book_df, movie_df)
print(f"Cross-Domain Recommendations for User1 (Books to Movies): {recommended_movies}")