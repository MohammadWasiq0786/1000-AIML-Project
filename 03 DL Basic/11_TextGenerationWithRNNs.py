"""
Project 91. Text generation with RNNs
Description:
Text generation with RNNs involves training a Recurrent Neural Network on a large text corpus so it can generate text one character (or word) at a time. In this project, we use a character-level RNN (LSTM-based) using Keras to learn from a text sample and generate new, similar text.

About:
📝 What This Project Demonstrates:
Builds a character-level RNN using LSTM to learn text patterns

Trains on a real-world book (e.g., Alice in Wonderland)

Generates new text sequences with temperature-controlled creativity
"""

# Install if not already: pip install tensorflow
 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
 
# Load sample text (e.g., Alice in Wonderland)
path = tf.keras.utils.get_file('alice.txt', origin='https://www.gutenberg.org/files/11/11-0.txt')
with open(path, 'r', encoding='utf-8') as f:
    text = f.read().lower()
 
# Reduce to a smaller sample for demo
text = text[:100000]
 
# Create character mappings
chars = sorted(set(text))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = np.array(chars)
 
# Vectorize the text
seq_length = 100
step = 1
sequences = []
next_chars = []
 
for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i:i + seq_length])
    next_chars.append(text[i + seq_length])
 
X = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.bool_)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool_)
 
for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t, char2idx[char]] = 1
    y[i, char2idx[next_chars[i]]] = 1
 
# Build RNN model
model = Sequential([
    LSTM(128, input_shape=(seq_length, len(chars))),
    Dense(len(chars), activation='softmax')
])
 
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, y, batch_size=128, epochs=5)
 
# Text generation function
def generate_text(model, seed, length=300, temperature=1.0):
    generated = seed
    input_seq = np.zeros((1, seq_length, len(chars)))
 
    for i, char in enumerate(seed):
        input_seq[0, i, char2idx[char]] = 1
 
    for _ in range(length):
        preds = model.predict(input_seq, verbose=0)[0]
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_idx = np.random.choice(len(chars), p=preds)
        next_char = idx2char[next_idx]
        generated += next_char
 
        # Update input sequence
        input_seq = np.roll(input_seq, -1, axis=1)
        input_seq[0, -1, :] = 0
        input_seq[0, -1, next_idx] = 1
 
    return generated
 
# Generate text using a seed
seed_text = text[:100]
generated = generate_text(model, seed=seed_text, length=500, temperature=0.5)
print("\nGenerated Text:\n")
print(generated)