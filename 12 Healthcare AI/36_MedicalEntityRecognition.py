"""
Project 476. Medical entity recognition
Description:
Medical Entity Recognition (MER) extracts structured medical concepts like diseases, drugs, symptoms, and procedures from unstructured clinical text. It is a key component in clinical NLP. In this project, we implement a simple named entity recognition (NER) pipeline using a pretrained BioNER model.

✅ What It Does:
Uses SciSpacy’s BioNER model to detect mentions of diseases, chemicals, and conditions.

Can extract terms like "hypertension", "diabetes", "metformin".

Extendable to:

Use custom NER models trained on i2b2, MIMIC-III, or PubMed

Extract entities like anatomical sites, tests, or body measurements

Feed into knowledge graphs, clinical search engines, or decision support systems
"""

# pip install scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz

import spacy
 
# 1. Load biomedical NER model (diseases + chemicals)
nlp = spacy.load("en_ner_bc5cdr_md")
 
# 2. Example clinical note
text = """
The patient was diagnosed with hypertension and type 2 diabetes. 
Treatment started with metformin and later added insulin. 
He also reported occasional dizziness and fatigue.
"""
 
# 3. Run NER
doc = nlp(text)
 
# 4. Extract and print medical entities
print("Detected Medical Entities:\n")
for ent in doc.ents:
    print(f"{ent.text} [{ent.label_}]")