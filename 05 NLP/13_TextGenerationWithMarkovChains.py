"""
Project 173. Text generation with Markov chains
Description:
Markov chains are a probabilistic model used to generate sequences based on the probability of transitions between states. In NLP, we can use them to generate realistic-looking text by predicting the next word (or character) based only on the current state. This project generates text by building a first-order Markov chain from a training corpus.

About:
🧠 What This Project Demonstrates:
Builds a first-order Markov chain to model word transitions

Generates new text sequences using random sampling

Captures local structure and style of the original corpus
"""

import random
import nltk
from collections import defaultdict
 
nltk.download('punkt')
 
# Sample corpus (can be replaced with a book or article)
corpus = """
Artificial intelligence is transforming how we work and live.
Machine learning is a subset of AI that enables systems to learn from data.
AI is used in healthcare, finance, and robotics.
The future of AI holds great promise and potential.
"""
 
# Step 1: Tokenize the corpus
words = nltk.word_tokenize(corpus.lower())
 
# Step 2: Build the Markov chain (word -> list of next possible words)
markov_chain = defaultdict(list)
for current_word, next_word in zip(words[:-1], words[1:]):
    markov_chain[current_word].append(next_word)
 
# Step 3: Generate text
def generate_text(chain, start_word, length=20):
    word = start_word
    output = [word]
 
    for _ in range(length - 1):
        next_words = chain.get(word, None)
        if not next_words:
            break  # stop if no next word
        word = random.choice(next_words)
        output.append(word)
    
    return ' '.join(output)
 
# Choose a starting word from the corpus
starting_word = "ai"
generated = generate_text(markov_chain, starting_word, length=25)
 
print("🧠 Generated Text with Markov Chain:\n")
print(generated)