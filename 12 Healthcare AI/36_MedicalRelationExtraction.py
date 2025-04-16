"""
Project 477. Medical relation extraction
Description:
Medical Relation Extraction (MRE) identifies relationships between medical entities in clinical text—such as which drug treats which disease or which symptom is associated with a condition. In this project, we'll use a rule-based approach to extract simple relations from clinical sentences, with potential for extension to deep learning models.

About:
✅ What It Does:
Uses NER to extract drugs and diseases, then matches textual patterns like “prescribed”, “developed” to extract relations.

Example output:

metformin —[treats]→ type 2 diabetes
nausea —[side_effect_of]→ metformin
Extendable to:

Use transformer-based models like BioBERT for RE

Build triplets for knowledge graphs

Handle complex relation types like causes, worsens, prevents
"""

# pip install scispacy
# # pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz

import spacy
 
# 1. Load a biomedical NER model (recognizes diseases and drugs)
nlp = spacy.load("en_ner_bc5cdr_md")
 
# 2. Sample clinical sentence
text = "The patient was prescribed metformin to manage type 2 diabetes and later developed nausea."
 
# 3. Run NER
doc = nlp(text)
 
# 4. Extract entities
entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
 
# 5. Identify simple relations (e.g., DRUG - treats -> DISEASE)
print("Detected Relations:\n")
for i in range(len(entities)):
    for j in range(i+1, len(entities)):
        ent1, label1, start1, end1 = entities[i]
        ent2, label2, start2, end2 = entities[j]
 
        # Check proximity and known pattern
        span = text[start1:end2] if start1 < end2 else text[start2:end1]
        if "prescribed" in span or "treat" in span or "manage" in span:
            if label1 == "CHEMICAL" and label2 == "DISEASE":
                print(f"{ent1} —[treats]→ {ent2}")
            elif label1 == "DISEASE" and label2 == "CHEMICAL":
                print(f"{ent2} —[treats]→ {ent1}")
        if "developed" in span and (label1 == "CHEMICAL" or label2 == "CHEMICAL"):
            print(f"{ent1} —[side_effect_of]→ {ent2}")