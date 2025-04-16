"""
Project 354. Course recommendation system
Description:
A course recommendation system suggests courses to users based on:

User learning history (e.g., topics previously studied, skill level)

Course content (e.g., course description, difficulty level)

In this project, we’ll build a course recommendation system that uses text similarity techniques (like TF-IDF or word embeddings) to recommend courses based on the user’s learning preferences and course descriptions.

About:
✅ What It Does:
Uses TF-IDF to convert course descriptions and the user profile into numerical features, capturing important keywords

Computes cosine similarity to measure how similar the user's interests are to each course

Recommends top courses based on content similarity between the user’s profile and course descriptions
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate course listings and course descriptions
courses = ['Python Programming', 'Data Science Bootcamp', 'Machine Learning Basics', 'Deep Learning Mastery', 'Web Development with Django']
course_descriptions = [
    "Learn the basics of Python programming including variables, data types, and control flow.",
    "An intensive bootcamp on data science concepts including data analysis, machine learning, and visualization.",
    "Introduction to machine learning, covering algorithms like regression, classification, and clustering.",
    "Master deep learning techniques including neural networks, CNNs, and RNNs for advanced AI applications.",
    "Learn how to develop full-stack web applications using Django, HTML, CSS, and JavaScript."
]
 
# 2. Simulate user learning history (topics they have studied or are interested in)
user_profile = "I have experience in Python programming and machine learning basics, and I am interested in deep learning."
 
# 3. Use TF-IDF to convert course descriptions and user profile into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(course_descriptions + [user_profile])  # Combine course descriptions and user profile
 
# 4. Function to recommend courses based on the user profile similarity to course descriptions
def course_recommendation(user_profile, courses, tfidf_matrix, top_n=3):
    # Compute the cosine similarity between the user profile and course descriptions
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get the indices of the most similar courses
    similar_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    recommended_courses = [courses[i] for i in similar_indices]
    return recommended_courses
 
# 5. Recommend courses based on the similarity to the user profile
recommended_courses = course_recommendation(user_profile, courses, tfidf_matrix)
print(f"Course Recommendations based on your profile: {recommended_courses}")