"""
Project 177. Word sense disambiguation
Description:
Word Sense Disambiguation is the process of identifying the correct meaning (sense) of a word in context when the word has multiple meanings. For example, “bank” could mean a financial institution or the side of a river. In this project, we implement a simple WSD system using the Lesk algorithm from NLTK, which disambiguates based on dictionary overlaps.

About:
🧠 What This Project Demonstrates:
Disambiguates ambiguous words in context using Lesk algorithm

Uses WordNet dictionary definitions to compare overlaps with sentence context

Outputs the correct meaning (sense) and its definition
"""

# Install if not already: pip install nltk
 
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
 
# Example sentences with ambiguous words
sentences = [
    "He went to the bank to deposit some money.",
    "The fisherman sat on the bank of the river.",
    "They need to book a room for the conference.",
    "She read a fascinating book about AI."
]
 
# Target ambiguous words
target_words = ['bank', 'bank', 'book', 'book']
 
# Apply Lesk algorithm to disambiguate word senses
print("🧠 Word Sense Disambiguation (Lesk Algorithm):\n")
for sentence, word in zip(sentences, target_words):
    context = word_tokenize(sentence)
    sense = lesk(context, word)
    print(f"📌 Sentence: {sentence}")
    if sense:
        print(f"🔍 Disambiguated Sense: {sense.name()} – {sense.definition()}\n")
    else:
        print("⚠️ Could not disambiguate the word.\n")