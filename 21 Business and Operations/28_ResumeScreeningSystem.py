"""
Project 828. Resume Screening System

A resume screening system automatically evaluates and filters resumes based on required skills or job descriptions. In this project, we simulate screening by extracting skills from resume text and matching them against job requirements using simple keyword matching and scoring.

This script evaluates resumes by counting keyword overlaps with the job's skill requirements. It provides a basic filter for recruiters to prioritize candidates. For advanced matching, use embeddings (e.g., SBERT), semantic similarity, or fine-tuned transformers for NLP-based screening.
"""

import pandas as pd
 
# Simulated resumes (text blocks)
resumes = [
    "Experienced software engineer skilled in Python, Java, and SQL. Worked on backend systems and APIs.",
    "Data analyst with expertise in Excel, Power BI, and SQL. Strong knowledge of data visualization.",
    "Full stack developer familiar with React, Node.js, MongoDB, and Python. Built web applications.",
    "Machine learning enthusiast with skills in Python, TensorFlow, and deep learning. Strong math background."
]
 
# Job requirement: list of required skills
required_skills = ['Python', 'SQL', 'APIs', 'TensorFlow']
 
# Function to score each resume by number of matching skills
def score_resume(text, skills):
    text_lower = text.lower()
    matches = [skill for skill in skills if skill.lower() in text_lower]
    return len(matches), matches
 
# Score all resumes
results = []
for idx, resume in enumerate(resumes):
    score, matched_skills = score_resume(resume, required_skills)
    results.append({
        'Candidate': f'Resume {idx+1}',
        'Score': score,
        'Matched Skills': ', '.join(matched_skills)
    })
 
# Create and display results table
df = pd.DataFrame(results).sort_values(by='Score', ascending=False)
print("Resume Screening Results:")
print(df)