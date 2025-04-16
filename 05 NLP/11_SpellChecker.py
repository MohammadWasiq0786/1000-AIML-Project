"""
Project 171. Spell checker implementation
Description:
A spell checker automatically detects and suggests corrections for misspelled words. In this project, we implement a simple spell checker using Levenshtein distance and word frequency from a known English dictionary. It's useful in search engines, writing tools, and input validation.

About:
🧠 What This Project Demonstrates:
Uses TextBlob’s spell correction powered by word probability + edit distance

Automatically fixes common spelling mistakes

Handles both individual words and entire sentences
"""

# Install if not already: pip install textblob
# And run once: python -m textblob.download_corpora
 
from textblob import TextBlob
 
# Example sentences with spelling mistakes
sentences = [
    "I realy like this librarry.",
    "Ths projct is relly amzing!",
    "Welcom to the wrld of NLP.",
]
 
print("🔍 Spell Checker Results:\n")
for sentence in sentences:
    blob = TextBlob(sentence)
    corrected = blob.correct()
    print(f"❌ Original:  {sentence}")
    print(f"✅ Corrected: {corrected}\n")