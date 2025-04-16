"""
Project 353. Job recommendation system
Description:
A job recommendation system suggests job listings to users based on:

User qualifications (e.g., skills, experience, education)

Job descriptions (e.g., required skills, experience, job role)

In this project, we’ll build a job recommendation system using text similarity techniques (such as TF-IDF or word embeddings) to recommend jobs based on the user's profile and job description.

About:
✅ What It Does:
Uses TF-IDF to convert job descriptions and user profile into numerical features, capturing important keywords

Computes cosine similarity to measure how similar the user's qualifications are to each job description

Recommends top jobs based on content similarity between the user’s profile and job descriptions
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulate job listings and job descriptions
jobs = ['Software Engineer', 'Data Scientist', 'Product Manager', 'UX Designer', 'Marketing Specialist']
job_descriptions = [
    "Develop software applications, write clean code, and work with other engineers.",
    "Analyze large datasets, build machine learning models, and conduct data analysis.",
    "Lead product development, define product strategy, and work with cross-functional teams.",
    "Design user experiences for web and mobile applications, conducting user research and testing.",
    "Create marketing campaigns, analyze market trends, and optimize digital marketing strategies."
]
 
# 2. Simulate user qualifications (skills, experience, etc.)
user_profile = "I have experience in software development, Python programming, and working with machine learning models."
 
# 3. Use TF-IDF to convert job descriptions and user profile into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(job_descriptions + [user_profile])  # Combine job descriptions and user profile
 
# 4. Function to recommend jobs based on the user profile similarity to job descriptions
def job_recommendation(user_profile, jobs, tfidf_matrix, top_n=3):
    # Compute the cosine similarity between the user profile and job descriptions
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get the indices of the most similar jobs
    similar_indices = cosine_sim.argsort()[-top_n:][::-1]
    
    recommended_jobs = [jobs[i] for i in similar_indices]
    return recommended_jobs
 
# 5. Recommend jobs based on the similarity to the user profile
recommended_jobs = job_recommendation(user_profile, jobs, tfidf_matrix)
print(f"Job Recommendations based on your profile: {recommended_jobs}")