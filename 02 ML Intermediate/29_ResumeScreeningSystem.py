"""
Project 69. Resume screening system
Description:
A resume screening system automates the evaluation of resumes to identify the most relevant candidates for a job. It compares resumes to job descriptions using text vectorization and cosine similarity. In this project, we simulate this process using TF-IDF and rank resumes by relevance to a job post.

About:
📄 What This Project Demonstrates:
Compares resumes to job descriptions using TF-IDF and cosine similarity

Ranks candidates based on semantic relevance

Builds a foundation for automated resume screening tools
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# Simulated job description
job_description = """
Looking for a Python developer with experience in machine learning, data analysis, and web development.
Familiarity with Django, pandas, and scikit-learn is preferred. Strong problem-solving and communication skills are essential.
"""
 
# Simulated resumes (could be parsed from PDFs in real use case)
resumes = [
    "Experienced Python developer with strong background in Django and web APIs. Built scalable applications.",
    "Data scientist skilled in Python, pandas, and machine learning. Worked on several classification problems.",
    "Frontend developer with React and JavaScript expertise. Limited experience with backend systems.",
    "Software engineer familiar with Java and Spring Boot. Looking to switch to data science roles.",
    "Machine learning enthusiast with academic projects in scikit-learn and deep learning using PyTorch."
]
 
# Combine job description with resumes for TF-IDF vectorization
documents = [job_description] + resumes
 
# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
 
# Compute cosine similarity between job description and each resume
job_vector = tfidf_matrix[0]
resume_vectors = tfidf_matrix[1:]
similarities = cosine_similarity(job_vector, resume_vectors).flatten()
 
# Rank resumes
ranked_resumes = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
 
# Display ranked results
print("Top Resume Matches:\n")
for idx, score in ranked_resumes:
    print(f"Resume {idx + 1} - Similarity Score: {score:.2f}")
    print(resumes[idx])
    print("-" * 60)