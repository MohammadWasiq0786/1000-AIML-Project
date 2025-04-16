"""
Project 89. Sentiment analysis with deep learning
Description:
Sentiment analysis is the task of determining the sentiment (positive, negative, or neutral) expressed in text. In this project, we build a deep learning model using Keras and an embedding layer + LSTM to classify movie reviews as positive or negative using the IMDb dataset.

About:
💬 What This Project Demonstrates:
Implements deep sentiment analysis using word embeddings + LSTM

Handles variable-length input via padding

Uses IMDb dataset, a popular benchmark for NLP
"""

# Install if not already: pip install tensorflow
 
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
 
# Load IMDb dataset (pre-tokenized, top 10,000 words)
vocab_size = 10000
max_len = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
 
# Pad sequences to uniform length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
 
# Build the deep learning model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    LSTM(64),
    Dense(1, activation='sigmoid')  # Binary classification: positive/negative
])
 
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
# Train model
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1)
 
# Evaluate model
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")
 
# Plot accuracy curve
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Sentiment Analysis - Deep Learning Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()