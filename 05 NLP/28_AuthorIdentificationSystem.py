"""
Project 188. Author identification system
Description:
An Author Identification System predicts the author of a text based on their unique writing style. This is useful in forensic linguistics, plagiarism detection, and literary analysis. In this project, we train a multi-class classifier using TF-IDF features and a Logistic Regression model to classify texts by author.

About:
🧠 What This Project Demonstrates:
Classifies text based on stylistic features (word choice, phrasing)

Uses TF-IDF vectors and a multi-class classifier

Predicts the most likely author of a given text
"""

# Install if not already: pip install scikit-learn pandas
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
 
# Sample dataset (texts labeled by authors)
data = {
    'text': [
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        "I celebrate myself, and sing myself, And what I assume you shall assume.",
        "The man in black fled across the desert, and the gunslinger followed.",
        "It was the best of times, it was the worst of times.",
        "Shall I compare thee to a summer’s day? Thou art more lovely and more temperate."
    ],
    'author': [
        "Jane Austen",
        "Walt Whitman",
        "Stephen King",
        "Charles Dickens",
        "William Shakespeare"
    ]
}
 
df = pd.DataFrame(data)
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['author'], test_size=0.4, random_state=42)
 
# Create TF-IDF + Logistic Regression pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])
 
# Train model
model.fit(X_train, y_train)
 
# Evaluate model
y_pred = model.predict(X_test)
print("🧠 Author Identification Report:\n")
print(classification_report(y_test, y_pred))
 
# Test on new sample
sample_text = ["O Romeo, Romeo! wherefore art thou Romeo?"]
predicted_author = model.predict(sample_text)[0]
print(f"\n🔍 Predicted Author for Sample Text:\n'{sample_text[0]}' → {predicted_author}")