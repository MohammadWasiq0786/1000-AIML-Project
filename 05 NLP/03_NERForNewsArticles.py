"""
Project 163. Named entity recognition for news articles
Description:
Named Entity Recognition (NER) identifies and classifies entities like people, organizations, locations, dates, etc., in text. In this project, we apply pretrained NER models (like SpaCy’s) to extract structured information from unstructured news article content.

About:
🧠 What This Project Demonstrates:
Performs Named Entity Recognition on unstructured news content

Extracts entities like ORG (organizations), GPE (locations), DATE, PERSON, etc.

Visualizes entity distribution using a word cloud
"""

# Install if not already: pip install spacy
# Then download SpaCy model: python -m spacy download en_core_web_sm
 
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
 
# Load SpaCy English NER model
nlp = spacy.load("en_core_web_sm")
 
# Sample news article text (can replace with real scraped data)
news_article = """
Apple Inc. is planning to open a new office in London by the end of 2025. 
According to CEO Tim Cook, the expansion is aimed at boosting their presence in Europe. 
The announcement was made during the London Tech Week.
"""
 
# Process text with SpaCy NER
doc = nlp(news_article)
 
# Extract named entities
print("🧠 Named Entities Found:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
 
# Optional: visualize entity labels (word cloud)
labels = [ent.label_ for ent in doc.ents]
label_freq = Counter(labels)
wc = WordCloud(background_color='white').generate_from_frequencies(label_freq)
 
plt.figure(figsize=(6, 4))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Entity Label Frequency (NER)")
plt.show()