"""
Project 458. Clinical trial matching system
Description:
A Clinical Trial Matching System automatically matches patients to relevant clinical trials based on diagnosis, genetic markers, location, or demographics. In this project, we simulate a text-based retrieval model using patient profiles and trial eligibility criteria, and compute a similarity score to suggest matching trials.

✅ What It Does:
Converts free-text patient data and trial eligibility into TF-IDF vectors.

Uses cosine similarity to rank the best clinical trial matches.

Can be extended to:

Use BERT embeddings (e.g., BioBERT, ClinicalBERT) for better semantic matching

Include structured filters (e.g., age, sex, biomarkers)

Create a recommendation dashboard for trial coordinators

You can later use real data from:

ClinicalTrials.gov API

UMLS + EHR structured data
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulated clinical trials and patient profile
trials = [
    {
        "title": "Lung Cancer Immunotherapy Trial",
        "criteria": "Adults with advanced non-small cell lung cancer and no prior immunotherapy treatment"
    },
    {
        "title": "Diabetes Type 2 Medication Study",
        "criteria": "Patients diagnosed with type 2 diabetes and HbA1c above 7.0%"
    },
    {
        "title": "Breast Cancer HER2+ Targeted Therapy",
        "criteria": "Female patients with HER2-positive breast cancer, stage II or III"
    },
    {
        "title": "COVID-19 Vaccine Booster Evaluation",
        "criteria": "Adults previously vaccinated with two doses and no recent infections"
    }
]
 
patient_profile = "65-year-old female with stage II HER2-positive breast cancer seeking targeted therapy"
 
# 2. Combine patient profile with trial criteria
corpus = [patient_profile] + [trial["criteria"] for trial in trials]
 
# 3. Compute TF-IDF embeddings
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
 
# 4. Compute cosine similarity between patient and each trial
similarities = cosine_similarity(X[0:1], X[1:]).flatten()
 
# 5. Rank and display top matches
sorted_indices = similarities.argsort()[::-1]
print("Top Clinical Trial Matches:\n")
for idx in sorted_indices:
    match_score = similarities[idx]
    print(f"- {trials[idx]['title']} (Similarity: {match_score:.2f})")