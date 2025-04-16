"""
Project 470. Personalized treatment recommendation
Description:
Personalized treatment recommendation systems use patient data to suggest the most effective therapies based on medical history, demographics, symptoms, and clinical guidelines. In this project, we simulate a basic recommendation engine using rule-based filtering and similarity matching between patient profiles and treatment plans.

✅ What It Does:
Matches a patient profile to best-fitting treatments using semantic similarity.

Uses TF-IDF and cosine similarity for recommendation logic.

Can be extended to:

Use clinical LLMs (e.g., BioGPT, ClinicalBERT)

Add structured data filters (e.g., allergies, renal function)

Integrate with FHIR APIs for real-time hospital deployment

For real-world applications:

Integrate with EHR systems, clinical decision support tools (CDSS), and NLP pipelines

Use clinical ontologies like SNOMED CT or ICD-10
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# 1. Simulated patient profile
patient_profile = "45-year-old male with type 2 diabetes and high blood pressure. BMI is 32. Needs medication for blood sugar control and weight management."
 
# 2. Simulated treatment plan database
treatments = [
    {
        "name": "Metformin + Lifestyle",
        "description": "Metformin for blood sugar control with diet and exercise plan. Recommended for overweight T2D patients."
    },
    {
        "name": "Insulin Therapy",
        "description": "Long-acting insulin for patients with uncontrolled blood glucose levels or contraindications to oral meds."
    },
    {
        "name": "SGLT2 Inhibitors",
        "description": "Improves blood sugar and helps reduce weight. Beneficial in obese T2D patients with cardiovascular risk."
    },
    {
        "name": "Dietary Coaching Only",
        "description": "Low-carb diet plan with weekly nutritionist follow-ups. Ideal for newly diagnosed T2D without complications."
    }
]
 
# 3. TF-IDF similarity for matching
corpus = [patient_profile] + [t["description"] for t in treatments]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
 
# 4. Compute similarities
sims = cosine_similarity(X[0:1], X[1:]).flatten()
ranked_indices = sims.argsort()[::-1]
 
# 5. Display recommendations
print("Top Treatment Recommendations:\n")
for idx in ranked_indices:
    print(f"- {treatments[idx]['name']} (Similarity: {sims[idx]:.2f})")
    print(f"  → {treatments[idx]['description']}\n")